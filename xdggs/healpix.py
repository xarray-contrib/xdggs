import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeVar

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import healpy
import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.grid import DGGSInfo
from xdggs.index import DGGSIndex
from xdggs.itertools import groupby, identity
from xdggs.utils import _extract_cell_id_variable, register_dggs

T = TypeVar("T")

try:
    ExceptionGroup
except NameError:  # pragma: no cover
    from exceptiongroup import ExceptionGroup


def polygons_shapely(vertices):
    import shapely

    return shapely.polygons(vertices)


def polygons_geoarrow(vertices):
    import pyproj
    from arro3.core import list_array

    polygon_vertices = np.concatenate([vertices, vertices[:, :1, :]], axis=1)
    crs = pyproj.CRS.from_epsg(4326)

    # construct geoarrow arrays
    coords = np.reshape(polygon_vertices, (-1, 2))
    coords_per_pixel = polygon_vertices.shape[1]
    geom_offsets = np.arange(vertices.shape[0] + 1, dtype="int32")
    ring_offsets = geom_offsets * coords_per_pixel

    polygon_array = list_array(geom_offsets, list_array(ring_offsets, coords))

    # We need to tag the array with extension metadata (`geoarrow.polygon`) so that Lonboard knows that this is a geospatial column.
    polygon_array_with_geo_meta = polygon_array.cast(
        polygon_array.field.with_metadata(
            {
                "ARROW:extension:name": "geoarrow.polygon",
                "ARROW:extension:metadata": json.dumps(
                    {"crs": crs.to_json_dict(), "edges": "spherical"}
                ),
            }
        )
    )
    return polygon_array_with_geo_meta


def center_around_prime_meridian(lon, lat):
    # three tasks:
    # - center around the prime meridian (map to a range of [-180, 180])
    # - replace the longitude of points at the poles with the median
    #   of longitude of the other vertices
    # - cells that cross the dateline should have longitudes around 180

    # center around prime meridian
    recentered = (lon + 180) % 360 - 180

    # replace lon of pole with the median of the remaining vertices
    contains_poles = np.isin(lat, np.array([-90, 90]))
    pole_cells = np.any(contains_poles, axis=-1)
    recentered[contains_poles] = np.median(
        np.reshape(
            recentered[pole_cells[:, None] & np.logical_not(contains_poles)], (-1, 3)
        ),
        axis=-1,
    )

    # keep cells that cross the dateline centered around 180
    polygons_to_fix = np.any(recentered < -100, axis=-1) & np.any(
        recentered > 100, axis=-1
    )
    result = np.where(
        polygons_to_fix[:, None] & (recentered < 0), recentered + 360, recentered
    )

    return result


@dataclass(frozen=True)
class HealpixInfo(DGGSInfo):
    resolution: int

    indexing_scheme: Literal["nested", "ring", "unique"] = "nested"

    valid_parameters: ClassVar[dict[str, Any]] = {
        "resolution": range(0, 29 + 1),
        "indexing_scheme": ["nested", "ring", "unique"],
    }

    def __post_init__(self):
        if self.resolution not in self.valid_parameters["resolution"]:
            raise ValueError("resolution must be an integer in the range of [0, 29]")

        if self.indexing_scheme not in self.valid_parameters["indexing_scheme"]:
            raise ValueError(
                f"indexing scheme must be one of {self.valid_parameters['indexing_scheme']}"
            )

    @property
    def nside(self: Self) -> int:
        return 2**self.resolution

    @property
    def nest(self: Self) -> bool:
        if self.indexing_scheme not in {"nested", "ring"}:
            raise ValueError(
                f"cannot convert indexing scheme {self.indexing_scheme} to `nest`"
            )
        else:
            return self.indexing_scheme == "nested"

    @classmethod
    def from_dict(cls: type[T], mapping: dict[str, Any]) -> T:
        def translate_nside(nside):
            log = np.log2(nside)
            potential_resolution = int(log)
            if potential_resolution != log:
                raise ValueError("`nside` has to be an integer power of 2")

            return potential_resolution

        translations = {
            "nside": ("resolution", translate_nside),
            "order": ("resolution", identity),
            "level": ("resolution", identity),
            "depth": ("resolution", identity),
            "nest": ("indexing_scheme", lambda nest: "nested" if nest else "ring"),
        }

        def translate(name, value):
            new_name, translator = translations.get(name, (name, identity))

            return new_name, name, translator(value)

        translated = (translate(name, value) for name, value in mapping.items())

        grouped = {
            name: [(old_name, value) for _, old_name, value in group]
            for name, group in groupby(translated, key=operator.itemgetter(0))
        }
        duplicated_parameters = {
            name: group for name, group in grouped.items() if len(group) != 1
        }
        if duplicated_parameters:
            raise ExceptionGroup(
                "received multiple values for parameters",
                [
                    ValueError(
                        f"Parameter {name} received multiple values: {sorted(n for n, _ in group)}"
                    )
                    for name, group in duplicated_parameters.items()
                ],
            )

        params = {
            name: group[0][1] for name, group in grouped.items() if name != "grid_name"
        }

        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        return {
            "grid_name": "healpix",
            "resolution": self.resolution,
            "indexing_scheme": self.indexing_scheme,
        }

    def cell_ids2geographic(self, cell_ids):
        lon, lat = healpy.pix2ang(self.nside, cell_ids, nest=self.nest, lonlat=True)

        return lon, lat

    def geographic2cell_ids(self, lon, lat):
        return healpy.ang2pix(self.nside, lon, lat, lonlat=True, nest=self.nest)

    def cell_boundaries(self, cell_ids: Any, backend="shapely") -> np.ndarray:
        boundary_vectors = healpy.boundaries(
            self.nside, cell_ids, step=1, nest=self.nest
        )

        lon, lat = healpy.vec2ang(np.moveaxis(boundary_vectors, 1, -1), lonlat=True)
        lon_reshaped = np.reshape(lon, (-1, 4))
        lat_reshaped = np.reshape(lat, (-1, 4))

        lon_ = center_around_prime_meridian(lon_reshaped, lat_reshaped)

        vertices = np.stack((lon_, lat_reshaped), axis=-1)

        backends = {
            "shapely": polygons_shapely,
            "geoarrow": polygons_geoarrow,
        }

        backend_func = backends.get(backend)
        if backend_func is None:
            raise ValueError("invalid backend: {backend!r}")

        return backend_func(vertices)


@register_dggs("healpix")
class HealpixIndex(DGGSIndex):
    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        grid_info: DGGSInfo,
    ):
        if not isinstance(grid_info, HealpixInfo):
            raise ValueError(f"grid info object has an invalid type: {type(grid_info)}")

        super().__init__(cell_ids, dim, grid_info)

    @classmethod
    def from_variables(
        cls: type["HealpixIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "HealpixIndex":
        _, var, dim = _extract_cell_id_variable(variables)

        grid_info = HealpixInfo.from_dict(var.attrs | options)

        return cls(var.data, dim, grid_info)

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(new_pd_index, self._dim, self._grid)

    @property
    def grid_info(self) -> HealpixInfo:
        return self._grid

    def _repr_inline_(self, max_width: int):
        return f"HealpixIndex(nside={self._grid.resolution}, indexing_scheme={self._grid.indexing_scheme})"
