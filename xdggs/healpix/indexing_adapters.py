from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from xarray.core.indexing import (
    ExplicitIndexer,
    IndexingAdapter,
    OuterIndexer,
    VectorizedIndexer,
)

from xdggs.healpix.grid_info import HealpixInfo

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable
    from typing import Any, Self

    from healpix_geo.nested import RangeMOCIndex


class MocRangesIndexingAdapter(IndexingAdapter):
    _index: RangeMOCIndex
    _grid_info: HealpixInfo
    _coord_name: Hashable
    _dims: tuple[str, ...]

    def __init__(
        self,
        index: RangeMOCIndex,
        grid_info: HealpixInfo,
        coord_name: Hashable,
        dims: tuple[str, ...] | None = None,
    ):
        self._index = index
        self._grid_info = grid_info

        self._coord_name = coord_name
        self._dims = dims

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint64)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._index.size,)

    @property
    def nbytes(self) -> int:
        return self._index.nbytes

    @property
    def _in_memory(self) -> bool:
        return False

    def _replace_data(self, index: RangeMOCIndex) -> Self:
        return type(self)(
            index,
            self._grid_info,
            self._coord_name,
            self._dims,
        )

    def get_duck_array(self) -> np.ndarray:
        return self._index.cell_ids()

    def _oindex_get(self, indexer: OuterIndexer) -> Self:
        if len(indexer.tuple) != 1:
            raise ValueError(
                f"MOC range array has exactly one dimension, but got indexer: {indexer}"
            )

        slice_ = indexer.tuple[0]
        return self._replace_data(self._index.isel(slice_))

    def _oindex_set(self, indexer: OuterIndexer, value: Any) -> None:
        raise TypeError("setting values is not supported on MOC range arrays")

    def _vindex_get(self, indexer: VectorizedIndexer):
        raise NotImplementedError("vindex_get not yet supported")

    def _vindex_set(self, indexer: VectorizedIndexer, value: Any) -> None:
        raise TypeError("setting values is not supported on MOC range arrays")

    def __getitem__(self, indexer: ExplicitIndexer):
        self._check_and_raise_if_non_basic_indexer(indexer)
        if len(indexer.tuple) != 1:
            raise ValueError("the array is one-dimensional")

        return self._oindex_get(OuterIndexer(indexer.tuple))

    def __setitem__(self, indexer: ExplicitIndexer, value: Any) -> None:
        raise TypeError("setting values is not supported on MOC range arrays")

    def transpose(self, order: Iterable[int]) -> Self:
        if len(order) != 1 or order[0] != 0:
            raise ValueError("axes don't match the 1-d array")

        # no transposition necessary for 1D arrays
        return self

    def __repr__(self: Any) -> str:
        return f"{type(self).__name__}(level={self._grid_info.level})"

    def _repr_inline_(self, max_width: int) -> str:
        # we want to display values in the inline repr for this lazy coordinate
        # `format_array_flat` prevents loading the whole array in memory.
        from xarray.core.formatting import format_array_flat

        return format_array_flat(self, max_width)
