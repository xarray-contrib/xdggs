from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr
from h3ronpy.arrow.vector import cells_to_coordinates, coordinates_to_cells
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs


@register_dggs("h3")
class H3Index(DGGSIndex):
    _resolution: int

    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        resolution: int,
    ):
        super().__init__(cell_ids, dim)
        self._resolution = resolution

    @classmethod
    def from_variables(
        cls: type["H3Index"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "H3Index":
        _, var, dim = _extract_cell_id_variable(variables)

        resolution = var.attrs.get("resolution", options.get("resolution"))
        return cls(var.data, dim, resolution)

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(new_pd_index, self._dim, self._resolution)

    def _latlon2cellid(self, lat: Any, lon: Any) -> np.ndarray:
        return coordinates_to_cells(lat, lon, self._resolution, radians=False)

    def _cellid2latlon(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        return cells_to_coordinates(cell_ids, radians=False)

    def _repr_inline_(self, max_width: int):
        return f"H3Index(resolution={self._resolution})"
