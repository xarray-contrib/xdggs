from collections.abc import Hashable, Mapping
from typing import Any, Union

import numpy as np
import xarray as xr
from xarray.indexes import Index, PandasIndex

from xdggs.grid import DGGSInfo
from xdggs.utils import GRID_REGISTRY, _extract_cell_id_variable


def decode(ds, grid_info=None, *, name="cell_ids"):
    """
    decode grid parameters and create a DGGS index

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset. Must contain a `"cell_ids"` coordinate with at least
        the attributes `grid_name` and `resolution`.

    Returns
    -------
    decoded : xarray.Dataset
        The input dataset with a DGGS index on the ``"cell_ids"`` coordinate.
    """

    return ds.dggs.decode(name=name, grid_info=grid_info)


class DGGSIndex(Index):
    _dim: str
    _pd_index: PandasIndex

    def __init__(self, cell_ids: Any | PandasIndex, dim: str, grid_info: DGGSInfo):
        self._dim = dim

        if isinstance(cell_ids, PandasIndex):
            self._pd_index = cell_ids
        else:
            self._pd_index = PandasIndex(cell_ids, dim)

        self._grid = grid_info

    @classmethod
    def from_variables(
        cls: type["DGGSIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "DGGSIndex":
        _, var, _ = _extract_cell_id_variable(variables)

        grid_name = var.attrs["grid_name"]
        cls = GRID_REGISTRY.get(grid_name)
        if cls is None:
            raise ValueError(f"unknown DGGS grid name: {grid_name}")

        return cls.from_variables(variables, options=options)

    def create_variables(
        self, variables: Mapping[Any, xr.Variable] | None = None
    ) -> dict[Hashable, xr.Variable]:
        return self._pd_index.create_variables(variables)

    def isel(
        self: "DGGSIndex", indexers: Mapping[Any, int | np.ndarray | xr.Variable]
    ) -> Union["DGGSIndex", None]:
        new_pd_index = self._pd_index.isel(indexers)
        if new_pd_index is not None:
            return self._replace(new_pd_index)
        else:
            return None

    def sel(self, labels, method=None, tolerance=None):
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        return self._pd_index.sel(labels, method=method, tolerance=tolerance)

    def _replace(self, new_pd_index: PandasIndex):
        raise NotImplementedError()

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        return self._grid.cell_ids2geographic(self._pd_index.index.values)

    def cell_boundaries(self) -> np.ndarray:
        return self.grid_info.cell_boundaries(self._pd_index.index.values)

    @property
    def grid_info(self) -> DGGSInfo:
        return self._grid
