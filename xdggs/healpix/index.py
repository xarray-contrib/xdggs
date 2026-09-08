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
        cell_ids: Any,
        *,
        dim: str,
        name: str,
        grid_info: DGGSInfo,
        index_kind: str | None = None,
        **options,
    ):
        self._dim = dim
        self._name = name
        self._grid = grid_info

        if not isinstance(grid_info, HealpixInfo):
            raise ValueError(f"grid info object has an invalid type: {type(grid_info)}")

        if isinstance(cell_ids, HealpixMocIndex):
            if index_kind not in ("moc", None):
                raise ValueError(
                    f"received a moc index instance but index_kind does not match: {index_kind}"
                )
            index_kind = "moc"
        elif index_kind is None:
            index_kind = "pandas"

        if index_kind == "pandas" and "compression" in options:
            raise ValueError("the pandas backend does not support compressed cell ids")

        self._kind = index_kind

        if isinstance(cell_ids, xr.Index):
            self._index = cell_ids
        elif index_kind == "pandas":
            self._index = PandasIndex(cell_ids, dim)
            self._index.index.name = name
        elif index_kind == "moc":
            self._index = HealpixMocIndex.from_array(
                cell_ids, dim=dim, grid_info=grid_info, name=name, **options
            )

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

        grid_info = HealpixInfo.from_dict(var.attrs)

        return cls(var.data, dim=dim, name=name, grid_info=grid_info, **options)

    def _replace(self, new_index: xr.Index):
        return type(self)(
            new_index,
            dim=self._dim,
            name=self._name,
            grid_info=self._grid,
            index_kind=self._kind,
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
