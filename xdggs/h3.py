import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import numpy as np
import xarray as xr
from h3ronpy.arrow.vector import (
    cells_to_coordinates,
    cells_to_wkb_polygons,
    coordinates_to_cells,
)
from xarray.indexes import PandasIndex

from xdggs.grid import DGGSInfo
from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs


def polygons_shapely(wkb):
    import shapely

    return shapely.from_wkb(wkb)


def polygons_geoarrow(wkb):
    import pyproj
    import shapely
    from arro3.core import list_array

    polygons = shapely.from_wkb(wkb)
    crs = pyproj.CRS.from_epsg(4326)

    geometry_type, coords, (ring_offsets, geom_offsets) = shapely.to_ragged_array(
        polygons
    )

    if geometry_type != shapely.GeometryType.POLYGON:
        raise ValueError(f"unsupported geometry type found: {geometry_type}")

    polygon_array = list_array(
        geom_offsets.astype("int32"), list_array(ring_offsets.astype("int32"), coords)
    )
    polygon_array_with_geo_meta = polygon_array.cast(
        polygon_array.field.with_metadata(
            {
                "ARROW:extension:name": "geoarrow.polygon",
                "ARROW:extension:metadata": json.dumps({"crs": crs.to_json_dict()}),
            }
        )
    )

    return polygon_array_with_geo_meta


@dataclass(frozen=True)
class H3Info(DGGSInfo):
    resolution: int

    valid_parameters: ClassVar[dict[str, Any]] = {"resolution": range(16)}

    def __post_init__(self):
        if self.resolution not in self.valid_parameters["resolution"]:
            raise ValueError("resolution must be an integer between 0 and 15")

    @classmethod
    def from_dict(cls: type[Self], mapping: dict[str, Any]) -> Self:
        params = {k: v for k, v in mapping.items() if k != "grid_name"}
        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        return {"grid_name": "h3", "resolution": self.resolution}

    def cell_ids2geographic(
        self, cell_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lat, lon = cells_to_coordinates(cell_ids, radians=False)

        return lon, lat

    def geographic2cell_ids(self, lon, lat):
        return coordinates_to_cells(lat, lon, self.resolution, radians=False)

    def cell_boundaries(self, cell_ids, backend="shapely"):
        # TODO: convert cell ids directly to geoarrow once h3ronpy supports it
        wkb = cells_to_wkb_polygons(cell_ids, radians=False, link_cells=False)

        backends = {
            "shapely": polygons_shapely,
            "geoarrow": polygons_geoarrow,
        }
        backend_func = backends.get(backend)
        if backend_func is None:
            raise ValueError("invalid backend: {backend!r}")
        return backend_func(wkb)


@register_dggs("h3")
class H3Index(DGGSIndex):
    _grid: DGGSInfo

    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        grid_info: DGGSInfo,
    ):
        super().__init__(cell_ids, dim, grid_info)

    @classmethod
    def from_variables(
        cls: type["H3Index"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "H3Index":
        _, var, dim = _extract_cell_id_variable(variables)

        grid_info = H3Info.from_dict(var.attrs | options)

        return cls(var.data, dim, grid_info)

    @property
    def grid_info(self) -> H3Info:
        return self._grid

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(new_pd_index, self._dim, self._grid)

    def _repr_inline_(self, max_width: int):
        return f"H3Index(resolution={self._grid.resolution})"
