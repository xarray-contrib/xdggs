from collections.abc import Mapping
from typing import Any, Self

import xarray as xr
from xarray.core.indexes import PandasIndex

from xdggs.grid import DGGSInfo
from xdggs.healpix.grid_info import HealpixInfo
from xdggs.healpix.moc_index import HealpixMocIndex
from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs


@register_dggs("healpix")
class HealpixIndex(DGGSIndex):
    def __init__(
        self,
        cell_ids: Any | xr.Index,
        dim: str,
        name: str,
        grid_info: DGGSInfo,
        index_kind: str = "pandas",
    ):
        if not isinstance(grid_info, HealpixInfo):
            raise ValueError(f"grid info object has an invalid type: {type(grid_info)}")

        self._dim = dim
        self._name = name

        if isinstance(cell_ids, xr.Index):
            self._index = cell_ids
        elif index_kind == "pandas":
            self._index = PandasIndex(cell_ids, dim)
            self._index.index.name = name
        elif index_kind == "moc":
            self._index = HealpixMocIndex.from_array(
                cell_ids, dim=dim, grid_info=grid_info, name=name
            )
        self._kind = index_kind

        self._grid = grid_info

    def values(self):
        if self._kind == "moc":
            return self._index._index.cell_ids()
        else:
            return self._index.index.values

    @classmethod
    def from_variables(
        cls: type[Self],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "HealpixIndex":
        name, var, dim = _extract_cell_id_variable(variables)

        index_kind = options.pop("index_kind", "pandas")

        grid_info = HealpixInfo.from_dict(var.attrs | options)

        return cls(var.data, dim, name, grid_info, index_kind=index_kind)

    def _replace(self, new_index: xr.Index):
        return type(self)(
            new_index, self._dim, self._name, self._grid, index_kind=self._kind
        )

    @property
    def grid_info(self) -> HealpixInfo:
        return self._grid

    def __repr__(self):
        return "\n".join(
            [
                f"<HealpixIndex(kind={self._kind})>",
                repr(self._grid),
            ]
        )

    def _repr_inline_(self, max_width: int):
        return f"HealpixIndex(level={self._grid.level}, indexing_scheme={self._grid.indexing_scheme}, kind={self._kind})"
