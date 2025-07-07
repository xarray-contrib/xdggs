import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeVar

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import cdshealpix.nested
import cdshealpix.ring
import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.grid import DGGSInfo, translate_parameters
from xdggs.index import DGGSIndex
from xdggs.itertools import identity
from xdggs.utils import _extract_cell_id_variable, register_dggs

T = TypeVar("T")


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
    """
    Grid information container for healpix grids.

    Parameters
    ----------
    level : int
        Grid hierarchical level. A higher value corresponds to a finer grid resolution
        with smaller cell areas. The number of cells covering the whole sphere usually
        grows exponentially with increasing level values, ranging from 5-100 cells at
        level 0 to millions or billions of cells at level 10+ (the exact numbers depends
        on the specific grid).
    indexing_scheme : {"nested", "ring", "unique"}, default: "nested"
        The indexing scheme of the healpix grid.

        .. warning::
            Note that ``"unique"`` is currently not supported as the underlying library
            (:doc:`cdshealpix <cdshealpix-python:index>`) does not support it.
    """

    level: int
    """int : The hierarchical level of the grid"""

    indexing_scheme: Literal["nested", "ring", "unique"] = "nested"
    """int : The indexing scheme of the grid"""

    valid_parameters: ClassVar[dict[str, Any]] = {
        "level": range(0, 29 + 1),
        "indexing_scheme": ["nested", "ring", "unique"],
    }

    def __post_init__(self):
        if self.level not in self.valid_parameters["level"]:
            raise ValueError("level must be an integer in the range of [0, 29]")

        if self.indexing_scheme not in self.valid_parameters["indexing_scheme"]:
            raise ValueError(
                f"indexing scheme must be one of {self.valid_parameters['indexing_scheme']}"
            )
        elif self.indexing_scheme == "unique":
            raise ValueError("the indexing scheme `unique` is currently not supported")

    @property
    def nside(self: Self) -> int:
        """resolution as the healpy-compatible nside parameter"""
        return 2**self.level

    @property
    def nest(self: Self) -> bool:
        """indexing_scheme as the healpy-compatible nest parameter"""
        if self.indexing_scheme not in {"nested", "ring"}:
            raise ValueError(
                f"cannot convert indexing scheme {self.indexing_scheme} to `nest`"
            )
        else:
            return self.indexing_scheme == "nested"

    @classmethod
    def from_dict(cls: type[T], mapping: dict[str, Any]) -> T:
        """construct a `HealpixInfo` object from a mapping of attributes

        Parameters
        ----------
        mapping: mapping of str to any
            The attributes.

        Returns
        -------
        grid_info : HealpixInfo
            The constructed grid info object.
        """

        def translate_nside(nside):
            log = np.log2(nside)
            potential_level = int(log)
            if potential_level != log:
                raise ValueError("`nside` has to be an integer power of 2")

            return potential_level

        translations = {
            "nside": ("level", translate_nside),
            "order": ("level", identity),
            "resolution": ("level", identity),
            "depth": ("level", identity),
            "nest": ("indexing_scheme", lambda nest: "nested" if nest else "ring"),
        }

        params = translate_parameters(mapping, translations)
        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Dump the normalized grid parameters.

        Returns
        -------
        mapping : dict of str to any
            The normalized grid parameters.
        """
        return {
            "grid_name": "healpix",
            "level": self.level,
            "indexing_scheme": self.indexing_scheme,
        }

    def cell_ids2geographic(self, cell_ids):
        """
        Convert cell ids to geographic coordinates

        Parameters
        ----------
        cell_ids : array-like
            Array-like containing the cell ids.

        Returns
        -------
        lon : array-like
            The longitude coordinate values of the grid cells in degree
        lat : array-like
            The latitude coordinate values of the grid cells in degree
        """
        converters = {
            "nested": cdshealpix.nested.healpix_to_lonlat,
            "ring": lambda cell_ids, level: cdshealpix.ring.healpix_to_lonlat(
                cell_ids, nside=2**level
            ),
        }
        converter = converters[self.indexing_scheme]

        lon, lat = converter(cell_ids, self.level)

        return np.asarray(lon.to("degree")), np.asarray(lat.to("degree"))

    def geographic2cell_ids(self, lon, lat):
        """
        Convert cell ids to geographic coordinates

        This will perform a binning operation: any point within a grid cell will be assign
        that cell's ID.

        Parameters
        ----------
        lon : array-like
            The longitude coordinate values in degree
        lat : array-like
            The latitude coordinate values in degree

        Returns
        -------
        cell_ids : array-like
            Array-like containing the cell ids.
        """
        from astropy.coordinates import Latitude, Longitude

        converters = {
            "nested": cdshealpix.nested.lonlat_to_healpix,
            "ring": lambda lon, lat, level: cdshealpix.ring.lonlat_to_healpix(
                lon, lat, nside=2**level
            ),
        }
        converter = converters[self.indexing_scheme]

        longitude = Longitude(lon, unit="degree")
        latitude = Latitude(lat, unit="degree")

        return converter(longitude, latitude, self.level)

    def cell_boundaries(self, cell_ids: Any, backend="shapely") -> np.ndarray:
        """
        Derive cell boundary polygons from cell ids

        Parameters
        ----------
        cell_ids : array-like
            The cell ids.
        backend : {"shapely", "geoarrow"}, default: "shapely"
            The backend to convert to.

        Returns
        -------
        polygons : array-like
            The derived cell boundary polygons. The format differs based on the passed
            backend:

            - ``"shapely"``: return a array of :py:class:`shapely.Polygon` objects
            - ``"geoarrow"``: return a ``geoarrow`` array
        """
        converters = {
            "nested": cdshealpix.nested.vertices,
            "ring": lambda cell_ids, level, **kwargs: cdshealpix.ring.vertices(
                cell_ids, nside=2**level, **kwargs
            ),
        }
        converter = converters[self.indexing_scheme]

        lon_, lat_ = converter(cell_ids, self.level, step=1)

        lon = np.asarray(lon_.to("degree"))
        lat = np.asarray(lat_.to("degree"))

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
            raise ValueError(f"invalid backend: {backend!r}")

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
        return f"HealpixIndex(level={self._grid.level}, indexing_scheme={self._grid.indexing_scheme})"
