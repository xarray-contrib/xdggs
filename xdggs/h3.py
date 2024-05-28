from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import numpy as np
import xarray as xr
from h3ronpy.arrow.vector import cells_to_coordinates, coordinates_to_cells
from xarray.indexes import PandasIndex

from xdggs.grid import DGGSInfo
from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs


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


@register_dggs("h3")
class H3Index(DGGSIndex):
    _grid: DGGSInfo

    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        grid_info: DGGSInfo,
    ):
        super().__init__(cell_ids, dim)
        self._grid = grid_info

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
    def grid(self) -> H3Info:
        return self._grid

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(new_pd_index, self._dim, self._grid)

    def _latlon2cellid(self, lat: Any, lon: Any) -> np.ndarray:
        return coordinates_to_cells(lat, lon, self._grid.resolution, radians=False)

    def _cellid2latlon(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        return cells_to_coordinates(cell_ids, radians=False)

    def _repr_inline_(self, max_width: int):
        return f"H3Index(resolution={self._grid.resolution})"
