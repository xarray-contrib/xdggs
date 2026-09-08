from dataclasses import dataclass

import numpy as np
import pytest
from healpix_geo.nested import RangeMOCIndex
from xarray.core.indexing import BasicIndexer, OuterIndexer

from xdggs.healpix import indexing_adapters
from xdggs.healpix.grid_info import HealpixInfo


@dataclass
class Container:
    index: RangeMOCIndex
    grid_info: HealpixInfo


@pytest.fixture
def index_grid():
    return Container(
        RangeMOCIndex.full_domain(5),
        grid_info=HealpixInfo(level=5, indexing_scheme="nested"),
    )


@pytest.fixture
def array(index_grid):
    return indexing_adapters.MocRangesIndexingAdapter(
        index_grid.index, index_grid.grid_info, "cell_ids", "cells"
    )


class TestMocRangesIndexingAdapter:
    @pytest.mark.parametrize("dim", ["cells", "cell", "zones"])
    @pytest.mark.parametrize("coord", ["cell_ids", "zonal_ids"])
    def test_init(self, index_grid, dim, coord):
        dims = (dim,)
        actual = indexing_adapters.MocRangesIndexingAdapter(
            index_grid.index, index_grid.grid_info, coord, dims
        )

        assert actual._index is index_grid.index
        assert actual._grid_info is index_grid.grid_info
        assert actual._dims is dims
        assert actual._coord_name is coord

    def test_dtype(self, array):
        assert array.dtype == np.dtype("uint64")

    def test_shape(self, array):
        assert array.shape == (12 * 4**array._grid_info.level,)

    def test_nbytes(self, array):
        assert array.nbytes == 16

    def test_in_memory(self, array):
        assert not array._in_memory

    def test_replace_data(self, array):
        new_index = array._index.isel(slice(500))
        actual = array._replace_data(new_index)

        assert actual._index is new_index
        assert actual.shape == (500,)

    def test_oindex_get(self, array):
        indexer = slice(20, 60)
        actual = array.oindex[OuterIndexer((indexer,))]
        expected = array._replace_data(array._index.isel(indexer))

        np.testing.assert_equal(actual._index.ranges(), expected._index.ranges())

    def test_getitem(self, array):
        indexer = slice(20, 60)
        actual = array[BasicIndexer((indexer,))]
        expected = array._replace_data(array._index.isel(indexer))

        np.testing.assert_equal(actual._index.ranges(), expected._index.ranges())
