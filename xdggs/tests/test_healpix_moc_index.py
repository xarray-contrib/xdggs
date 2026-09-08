import numpy as np
import pytest
import xarray as xr

from xdggs import healpix
from xdggs.tests import da, raise_if_dask_computes, requires_dask


class TestHealpixMocIndex:
    @pytest.mark.parametrize(
        ["level", "cell_ids", "max_computes"],
        (
            pytest.param(
                2, np.arange(12 * 4**2, dtype="uint64"), 1, id="numpy-2-full_domain"
            ),
            pytest.param(
                2,
                np.arange(3 * 4**2, 5 * 4**2, dtype="uint64"),
                1,
                id="numpy-2-region",
            ),
            pytest.param(
                10,
                da.arange(12 * 4**10, chunks=(4**6,), dtype="uint64"),
                0,
                marks=requires_dask,
                id="dask-10-full_domain",
            ),
            pytest.param(
                15,
                da.arange(12 * 4**15, chunks=(4**10,), dtype="uint64"),
                0,
                marks=requires_dask,
                id="dask-15-full_domain",
            ),
            pytest.param(
                10,
                da.arange(3 * 4**10, 5 * 4**10, chunks=(4**6,), dtype="uint64"),
                1,
                marks=requires_dask,
                id="dask-10-region",
            ),
        ),
    )
    def test_from_array(self, level, cell_ids, max_computes):
        grid_info = healpix.HealpixInfo(level=level, indexing_scheme="nested")

        with raise_if_dask_computes(max_computes=max_computes):
            index = healpix.HealpixMocIndex.from_array(
                cell_ids, dim="cells", name="cell_ids", grid_info=grid_info
            )

        assert isinstance(index, healpix.HealpixMocIndex)
        chunks = index.chunksizes["cells"]
        assert chunks is None or isinstance(chunks[0], int)
        assert index.size == cell_ids.size
        assert index.nbytes == 16

    def test_from_array_unsupported_indexing_scheme(self):
        level = 1
        cell_ids = np.arange(12 * 4**level, dtype="uint64")
        grid_info = healpix.HealpixInfo(level=level, indexing_scheme="ring")

        with pytest.raises(ValueError, match=".*only supports the 'nested' scheme"):
            healpix.HealpixMocIndex.from_array(
                cell_ids, dim="cells", name="cell_ids", grid_info=grid_info
            )

    @pytest.mark.parametrize("dask", [False, pytest.param(True, marks=requires_dask)])
    @pytest.mark.parametrize(
        ["level", "cell_ids"],
        (
            (
                1,
                np.array(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 23, 24, 25, 43, 45, 46, 47],
                    dtype="uint64",
                ),
            ),
            (4, np.arange(12 * 4**4, dtype="uint64")),
        ),
    )
    def test_from_variables(self, level, cell_ids, dask):
        grid_info_mapping = {
            "grid_name": "healpix",
            "level": level,
            "indexing_scheme": "nested",
        }
        variables = {"cell_ids": xr.Variable("cells", cell_ids, grid_info_mapping)}
        if dask:
            variables["cell_ids"] = variables["cell_ids"].chunk(4**level)

        actual = healpix.HealpixMocIndex.from_variables(variables, options={})

        assert isinstance(actual, healpix.HealpixMocIndex)
        assert actual.size == cell_ids.size
        np.testing.assert_equal(actual._index.cell_ids(), cell_ids)

    @pytest.mark.parametrize(
        "indexer",
        (
            slice(None),
            slice(None, 4**1),
            slice(2 * 4**1, 7 * 4**1),
            slice(7, 25),
            np.array([-4, -3, -2], dtype="int64"),
            np.array([12, 13, 14, 15, 16], dtype="uint64"),
            np.array([1, 2, 3, 4, 5], dtype="uint32"),
        ),
    )
    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param(None, id="none"),
            pytest.param((12, 12, 12, 12), marks=requires_dask, id="equally_sized"),
        ],
    )
    def test_isel(self, indexer, chunks):
        from healpix_geo.nested import RangeMOCIndex

        grid_info = healpix.HealpixInfo(level=1, indexing_scheme="nested")
        cell_ids = np.arange(12 * 4**grid_info.level, dtype="uint64")
        if chunks is None:
            input_chunks = None
            expected_chunks = None
        else:
            import dask.array as da

            cell_ids_ = da.arange(
                12 * 4**grid_info.level, dtype="uint64", chunks=chunks
            )
            input_chunks = cell_ids_.chunks[0]
            expected_chunks = cell_ids_[indexer].chunks[0]

        index = healpix.HealpixMocIndex(
            RangeMOCIndex.from_cell_ids(grid_info.level, cell_ids),
            dim="cells",
            name="cell_ids",
            grid_info=grid_info,
            chunksizes={"cells": input_chunks},
        )

        actual = index.isel({"cells": indexer})
        expected = healpix.HealpixMocIndex(
            RangeMOCIndex.from_cell_ids(grid_info.level, cell_ids[indexer]),
            dim="cells",
            name="cell_ids",
            grid_info=grid_info,
            chunksizes={"cells": expected_chunks},
        )

        assert isinstance(actual, healpix.HealpixMocIndex)
        assert actual.nbytes == expected.nbytes
        assert actual.chunksizes == expected.chunksizes
        np.testing.assert_equal(actual._index.cell_ids(), expected._index.cell_ids())

    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param((12, 12, 12, 12), marks=requires_dask),
            pytest.param((18, 10, 10, 10), marks=requires_dask),
            pytest.param((8, 12, 14, 14), marks=requires_dask),
            None,
        ],
    )
    def test_create_variables(self, chunks):
        from healpix_geo.nested import RangeMOCIndex

        grid_info = healpix.HealpixInfo(level=1, indexing_scheme="nested")
        cell_ids = np.arange(12 * 4**grid_info.level, dtype="uint64")
        indexer = slice(3 * 4**grid_info.level, 7 * 4**grid_info.level)
        index = healpix.HealpixMocIndex(
            RangeMOCIndex.from_cell_ids(grid_info.level, cell_ids[indexer]),
            dim="cells",
            name="cell_ids",
            grid_info=grid_info,
            chunksizes={"cells": chunks},
        )

        if chunks is not None:
            variables = {
                "cell_ids": xr.Variable("cells", cell_ids, grid_info.to_dict()).chunk(
                    {"cells": chunks}
                )
            }
        else:
            variables = {
                "cell_ids": xr.Variable("cells", cell_ids, grid_info.to_dict())
            }

        actual = index.create_variables(variables)
        expected = {"cell_ids": variables["cell_ids"].isel(cells=indexer)}

        assert actual.keys() == expected.keys()
        xr.testing.assert_equal(actual["cell_ids"], expected["cell_ids"])
        assert not actual["cell_ids"]._in_memory

    def test_create_variables_new(self):
        from healpix_geo.nested import RangeMOCIndex

        grid_info = healpix.HealpixInfo(level=1, indexing_scheme="nested")
        cell_ids = np.arange(12 * 4**grid_info.level, dtype="uint64")
        indexer = slice(3 * 4**grid_info.level, 7 * 4**grid_info.level)
        index = healpix.HealpixMocIndex(
            RangeMOCIndex.from_cell_ids(grid_info.level, cell_ids[indexer]),
            dim="cells",
            name="cell_ids",
            grid_info=grid_info,
            chunksizes={"cells": None},
        )
        actual = index.create_variables({})
        expected = {"cell_ids": xr.Variable("cells", cell_ids[indexer])}

        assert actual.keys() == expected.keys()
        xr.testing.assert_equal(actual["cell_ids"], expected["cell_ids"])
        assert not actual["cell_ids"]._in_memory

    @pytest.mark.parametrize(
        "indexer",
        (
            slice(None),
            slice(None, 4**1),
            slice(2 * 4**1, 7 * 4**1),
            slice(7, 25),
            np.array([12, 13, 14, 15, 16], dtype="uint64"),
            np.array([1, 2, 3, 4, 5], dtype="uint32"),
        ),
    )
    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param(None, id="none"),
            pytest.param((12, 12, 12, 12), marks=requires_dask, id="equally_sized"),
        ],
    )
    def test_sel(self, indexer, chunks):
        from healpix_geo.nested import RangeMOCIndex

        grid_info = healpix.HealpixInfo(level=1, indexing_scheme="nested")
        cell_ids = np.arange(12 * 4**grid_info.level, dtype="uint64")

        if isinstance(indexer, slice):
            start, stop, step = indexer.indices(cell_ids.size)
            if stop < cell_ids.size:
                stop += 1

            expected_indexer = slice(start, stop, step)
        else:
            expected_indexer = indexer

        if chunks is None:
            input_chunks = None
            expected_chunks = None
        else:
            import dask.array as da

            cell_ids_ = da.arange(
                12 * 4**grid_info.level, dtype="uint64", chunks=chunks
            )
            input_chunks = cell_ids_.chunks[0]
            expected_chunks = cell_ids_[expected_indexer].chunks[0]

        index = healpix.HealpixMocIndex(
            RangeMOCIndex.from_cell_ids(grid_info.level, cell_ids),
            dim="cells",
            name="cell_ids",
            grid_info=grid_info,
            chunksizes={"cells": input_chunks},
        )

        result = index.sel({"cell_ids": indexer})
        actual = result.indexes["cell_ids"]
        actual_indexer = result.dim_indexers["cells"]

        expected = healpix.HealpixMocIndex(
            RangeMOCIndex.from_cell_ids(grid_info.level, cell_ids[expected_indexer]),
            dim="cells",
            name="cell_ids",
            grid_info=grid_info,
            chunksizes={"cells": expected_chunks},
        )

        if isinstance(actual_indexer, slice):
            assert actual_indexer == expected_indexer
        else:
            np.testing.assert_equal(actual_indexer, expected_indexer)

        assert isinstance(actual, healpix.HealpixMocIndex)
        assert actual.nbytes == expected.nbytes
        assert actual.chunksizes == expected.chunksizes
        np.testing.assert_equal(actual._index.cell_ids(), expected._index.cell_ids())

    def test_sel_error(self):
        from healpix_geo.nested import RangeMOCIndex

        grid_info = healpix.HealpixInfo(level=1, indexing_scheme="nested")
        cell_ids = np.arange(12 * 4**grid_info.level, dtype="uint64")

        index = healpix.HealpixMocIndex(
            RangeMOCIndex.from_cell_ids(grid_info.level, cell_ids),
            dim="cells",
            name="cell_ids",
            grid_info=grid_info,
            chunksizes={"cells": None},
        )

        indexer = np.array([-4, 2, 1], dtype="int64")

        with pytest.raises(ValueError, match="Cell ids can't be negative"):
            index.sel({"cell_ids": indexer})

    def test_sel_kwargs(self):
        from healpix_geo.nested import RangeMOCIndex

        grid_info = healpix.HealpixInfo(level=1, indexing_scheme="nested")
        cell_ids = np.arange(12 * 4**grid_info.level, dtype="uint64")

        index = healpix.HealpixMocIndex(
            RangeMOCIndex.from_cell_ids(grid_info.level, cell_ids),
            dim="cells",
            name="cell_ids",
            grid_info=grid_info,
            chunksizes={"cells": None},
        )

        indexer = np.array([2, 1], dtype="uint64")

        # method is ignored by the moc index
        index.sel({"cell_ids": indexer}, method="unknown")

        with pytest.raises(TypeError):
            index.sel({"cell_ids": indexer}, tolerance=0.1)
