from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import numpy as np
import shapely
import xarray as xr
from h3ronpy.arrow.vector import (
    cells_to_coordinates,
    cells_to_wkb_polygons,
    coordinates_to_cells,
)
from xarray.indexes import PandasIndex

from xdggs.grid import DGGSInfo, translate_parameters
from xdggs.index import DGGSIndex
from xdggs.itertools import identity
from xdggs.utils import _extract_cell_id_variable, register_dggs


@dataclass(frozen=True)
class H3Info(DGGSInfo):
    level: int

    valid_parameters: ClassVar[dict[str, Any]] = {"level": range(16)}

    def __post_init__(self):
        if self.level not in self.valid_parameters["level"]:
            raise ValueError("level must be an integer between 0 and 15")

    @classmethod
    def from_dict(cls: type[Self], mapping: dict[str, Any]) -> Self:
        translations = {
            "resolution": ("level", identity),
        }

        params = translate_parameters(mapping, translations)
        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        return {"grid_name": "h3", "level": self.level}

    def cell_ids2geographic(
        self, cell_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lat, lon = cells_to_coordinates(cell_ids, radians=False)

        return lon, lat

    def geographic2cell_ids(self, lon, lat):
        return coordinates_to_cells(lat, lon, self.level, radians=False)

    def cell_boundaries(self, cell_ids):
        wkb = cells_to_wkb_polygons(cell_ids, radians=False, link_cells=False)

        return shapely.from_wkb(wkb)


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
        return f"H3Index(level={self._grid.level})"
