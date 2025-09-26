from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from xarray.indexes import Index, PandasIndex

from xdggs.grid import DGGSInfo
from xdggs.utils import GRID_REGISTRY, _extract_cell_id_variable

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping
    from typing import Any, Self

    from xarray.core.types import JoinOptions


def decode(ds, grid_info=None, *, name="cell_ids", index_options=None, **index_kwargs):
    """
    decode grid parameters and create a DGGS index

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset. Must contain a coordinate for the cell ids with at
        least the attributes `grid_name` and `level`.
    grid_info : dict or DGGSInfo, optional
        Override the grid parameters on the dataset. Useful to set attributes on
        the dataset.
    name : str, default: "cell_ids"
        The name of the coordinate containing the cell ids.
    index_options, **index_kwargs : dict, optional
        Additional options to forward to the index.

    Returns
    -------
    decoded : xarray.DataArray or xarray.Dataset
        The input dataset with a DGGS index on the cell id coordinate.

    See Also
    --------
    xarray.Dataset.dggs.decode
    xarray.DataArray.dggs.decode
    """
    if index_options is None:
        index_options = {}

    return ds.dggs.decode(
        name=name, grid_info=grid_info, index_options=index_options | index_kwargs
    )


class DGGSIndex(Index):
    _dim: str
    _index: xr.Index

    def __init__(self, cell_ids: Any | xr.Index, dim: str, grid_info: DGGSInfo):
        self._dim = dim

        if isinstance(cell_ids, xr.Index):
            self._index = cell_ids
        else:
            self._index = PandasIndex(cell_ids, dim)

        self._grid = grid_info

    @classmethod
    def from_variables(
        cls: type[DGGSIndex],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> DGGSIndex:
        name, var, _ = _extract_cell_id_variable(variables)

        grid_name = var.attrs["grid_name"]
        cls = GRID_REGISTRY.get(grid_name)
        if cls is None:
            raise ValueError(f"unknown DGGS grid name: {grid_name}")

        index = cls.from_variables(variables, options=options)
        if isinstance(index._index, PandasIndex):
            index._index.index.name = name

        return index

    def values(self):
        return self._index.index.values

    def equals(self, other: Index, **kwargs) -> bool:
        if (
            type(self) is not type(other)
            or self._dim != other._dim
            or self._grid != other._grid
        ):
            return False

        return self._index.equals(other._index, **kwargs)

    def create_variables(
        self, variables: Mapping[Any, xr.Variable] | None = None
    ) -> dict[Hashable, xr.Variable]:
        return self._index.create_variables(variables)

    def isel(
        self: DGGSIndex, indexers: Mapping[Any, int | np.ndarray | xr.Variable]
    ) -> DGGSIndex | None:
        new_index = self._index.isel(indexers)
        if new_index is not None:
            return self._replace(new_index)
        else:
            return None

    def sel(self, labels, method=None, tolerance=None):
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        return self._index.sel(labels, method=method, tolerance=tolerance)

    def join(self, other: Self, how: JoinOptions = "inner") -> Self:
        if self.grid_info != other.grid_info:
            raise ValueError(
                "Alignment with different grid parameters is not supported."
            )

        return self._replace(self._index.join(other._index, how=how))

    def reindex_like(self, other: Self) -> dict[Hashable, Any]:
        if self.grid_info != other.grid_info:
            raise ValueError(
                "Reindexing to different grid parameters is not supported."
            )

        return self._index.reindex_like(other._index)

    def _replace(self, new_index: PandasIndex):
        raise NotImplementedError()

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        return self._grid.cell_ids2geographic(self.values())

    def cell_boundaries(self) -> np.ndarray:
        return self.grid_info.cell_boundaries(self.values())

    def zoom_to(self, level: int) -> np.ndarray:
        return self._grid.zoom_to(self.values(), level=level)

    @property
    def grid_info(self) -> DGGSInfo:
        return self._grid
