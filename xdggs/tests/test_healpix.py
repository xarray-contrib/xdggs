import itertools

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
import shapely
import shapely.testing
import xarray as xr
import xarray.testing.strategies as xrst
from hypothesis import given
from xarray.core.indexes import PandasIndex

from xdggs import healpix
from xdggs.tests import (
    assert_exceptions_equal,
    da,
    geoarrow_to_shapely,
    raise_if_dask_computes,
    requires_dask,
)


# namespace class
class strategies:
    invalid_levels = st.integers(max_value=-1) | st.integers(min_value=30)
    levels = st.integers(min_value=0, max_value=29)
    # TODO: add back `"unique"` once that is supported
    indexing_schemes = st.sampled_from(["nested", "ring"])
    invalid_indexing_schemes = st.text().filter(lambda x: x not in ["nested", "ring"])

    dims = xrst.names()

    @classmethod
    def grid_mappings(cls):
        strategies = {
            "resolution": cls.levels,
            "nside": cls.levels.map(lambda n: 2**n),
            "depth": cls.levels,
            "level": cls.levels,
            "order": cls.levels,
            "indexing_scheme": cls.indexing_schemes,
            "nest": st.booleans(),
        }

        names = {
            "level": st.sampled_from(
                ["resolution", "nside", "depth", "level", "order"]
            ),
            "indexing_scheme": st.sampled_from(["indexing_scheme", "nest"]),
        }

        return st.builds(lambda **x: list(x.values()), **names).flatmap(
            lambda params: st.builds(dict, **{p: strategies[p] for p in params})
        )

    def cell_ids(max_value=None, dtypes=None):
        if dtypes is None:
            dtypes = st.sampled_from(["int32", "int64", "uint32", "uint64"])
        shapes = npst.array_shapes(min_dims=1, max_dims=1)

        return npst.arrays(
            dtypes,
            shapes,
            elements={"min_value": 0, "max_value": max_value},
            unique=True,
            fill=st.nothing(),
        )

    options = st.just({})

    def grids(
        levels=levels,
        indexing_schemes=indexing_schemes,
    ):
        return st.builds(
            healpix.HealpixInfo,
            level=levels,
            indexing_scheme=indexing_schemes,
        )

    @classmethod
    def grid_and_cell_ids(
        cls,
        levels=levels,
        indexing_schemes=indexing_schemes,
        dtypes=None,
    ):
        cell_levels = st.shared(levels, key="common-levels")
        grid_levels = st.shared(levels, key="common-levels")
        cell_ids_ = cell_levels.flatmap(
            lambda level: cls.cell_ids(max_value=12 * 4**level - 1, dtypes=dtypes)
        )
        grids_ = cls.grids(
            levels=grid_levels,
            indexing_schemes=indexing_schemes,
        )

        return cell_ids_, grids_


options = [{}]
variable_names = ["cell_ids", "zonal_ids", "zone_ids"]
variables = [
    xr.Variable(
        "cells",
        np.array([3]),
        {
            "grid_name": "healpix",
            "level": 0,
            "indexing_scheme": "nested",
        },
    ),
    xr.Variable(
        "zones",
        np.array([3]),
        {
            "grid_name": "healpix",
            "level": 0,
            "indexing_scheme": "ring",
        },
    ),
    xr.Variable(
        "cells",
        np.array([5, 11, 21]),
        {
            "grid_name": "healpix",
            "level": 1,
            "indexing_scheme": "nested",
        },
    ),
    xr.Variable(
        "zones",
        np.array([54, 70, 82, 91]),
        {
            "grid_name": "healpix",
            "level": 3,
            "indexing_scheme": "nested",
        },
    ),
]
variable_combinations = list(itertools.product(variables, repeat=2))


class TestHealpixInfo:
    @given(strategies.invalid_levels)
    def test_init_invalid_levels(self, level):
        with pytest.raises(
            ValueError, match="level must be an integer in the range of"
        ):
            healpix.HealpixInfo(level=level)

    @given(strategies.invalid_indexing_schemes)
    def test_init_invalid_indexing_scheme(self, indexing_scheme):
        with pytest.raises(ValueError, match="indexing scheme must be one of"):
            healpix.HealpixInfo(
                level=0,
                indexing_scheme=indexing_scheme,
            )

    @given(strategies.levels, strategies.indexing_schemes)
    def test_init(self, level, indexing_scheme):
        grid = healpix.HealpixInfo(level=level, indexing_scheme=indexing_scheme)

        assert grid.level == level
        assert grid.indexing_scheme == indexing_scheme

    @given(strategies.levels)
    def test_nside(self, level):
        grid = healpix.HealpixInfo(level=level)

        assert grid.nside == 2**level

    @given(strategies.indexing_schemes)
    def test_nest(self, indexing_scheme):
        grid = healpix.HealpixInfo(level=1, indexing_scheme=indexing_scheme)
        if indexing_scheme not in {"nested", "ring"}:
            with pytest.raises(
                ValueError, match="cannot convert indexing scheme .* to `nest`"
            ):
                grid.nest
            return

        expected = indexing_scheme == "nested"

        assert grid.nest == expected

    @given(strategies.grid_mappings())
    def test_from_dict(self, mapping) -> None:
        healpix.HealpixInfo.from_dict(mapping)

    @given(strategies.levels, strategies.indexing_schemes)
    def test_to_dict(self, level, indexing_scheme) -> None:
        grid = healpix.HealpixInfo(level=level, indexing_scheme=indexing_scheme)
        actual = grid.to_dict()

        assert set(actual) == {"grid_name", "level", "indexing_scheme"}
        assert actual["grid_name"] == "healpix"
        assert actual["level"] == level
        assert actual["indexing_scheme"] == indexing_scheme

    @given(strategies.levels, strategies.indexing_schemes)
    def test_roundtrip(self, level, indexing_scheme):
        mapping = {
            "grid_name": "healpix",
            "level": level,
            "indexing_scheme": indexing_scheme,
        }

        grid = healpix.HealpixInfo.from_dict(mapping)
        roundtripped = grid.to_dict()

        assert roundtripped == mapping

    @pytest.mark.parametrize(
        ["params", "cell_ids", "expected_coords"],
        (
            (
                {"level": 0, "indexing_scheme": "nested"},
                np.array([2]),
                np.array(
                    [
                        [-135.0, 0.0],
                        [-90.0, 41.8103149],
                        [-135.0, 90.0],
                        [-180.0, 41.8103149],
                    ]
                ),
            ),
            (
                {"level": 2, "indexing_scheme": "ring"},
                np.array([12, 54]),
                np.array(
                    [
                        [
                            [22.5, 41.8103149],
                            [30.0, 54.3409123],
                            [0.0, 66.44353569],
                            [0.0, 54.3409123],
                        ],
                        [
                            [-45.0, 19.47122063],
                            [-33.75, 30.0],
                            [-45.0, 41.8103149],
                            [-56.25, 30.0],
                        ],
                    ]
                ),
            ),
            (
                {"level": 3, "indexing_scheme": "nested"},
                np.array([293, 17]),
                np.array(
                    [
                        [
                            [-5.625, -4.78019185],
                            [0.0, 0.0],
                            [-5.625, 4.78019185],
                            [-11.25, 0.0],
                        ],
                        [
                            [73.125, 24.62431835],
                            [78.75, 30.0],
                            [73.125, 35.68533471],
                            [67.5, 30.0],
                        ],
                    ]
                ),
            ),
            (
                {"level": 2, "indexing_scheme": "nested"},
                np.array([79]),
                np.array(
                    [
                        [0.0, 19.47122063],
                        [11.25, 30],
                        [0.0, 41.8103149],
                        [-11.25, 30],
                    ]
                ),
            ),
        ),
    )
    @pytest.mark.parametrize("backend", ["shapely", "geoarrow"])
    def test_cell_boundaries(self, params, cell_ids, backend, expected_coords):
        grid = healpix.HealpixInfo.from_dict(params)

        actual = grid.cell_boundaries(cell_ids, backend=backend)

        backends = {
            "shapely": lambda arr: arr,
            "geoarrow": geoarrow_to_shapely,
        }
        converter = backends[backend]
        expected = shapely.polygons(expected_coords)

        shapely.testing.assert_geometries_equal(converter(actual), expected)

    @given(
        *strategies.grid_and_cell_ids(
            # a dtype casting bug in the valid range check of `cdshealpix`
            # causes this test to fail for large levels
            levels=st.integers(min_value=0, max_value=10),
            indexing_schemes=st.sampled_from(["nested", "ring"]),
            dtypes=st.sampled_from(["int64"]),
        )
    )
    def test_cell_center_roundtrip(self, cell_ids, grid) -> None:
        centers = grid.cell_ids2geographic(cell_ids)

        roundtripped = grid.geographic2cell_ids(lat=centers[1], lon=centers[0])

        np.testing.assert_equal(roundtripped, cell_ids)

    @pytest.mark.parametrize(
        ["cell_ids", "level", "indexing_scheme", "expected"],
        (
            pytest.param(
                np.array([3]),
                1,
                "ring",
                (np.array([315.0]), np.array([66.44353569089877])),
            ),
            pytest.param(
                np.array([5, 11, 21]),
                3,
                "nested",
                (
                    np.array([61.875, 33.75, 84.375]),
                    np.array([19.47122063, 24.62431835, 41.8103149]),
                ),
            ),
        ),
    )
    def test_cell_ids2geographic(
        self, cell_ids, level, indexing_scheme, expected
    ) -> None:
        grid = healpix.HealpixInfo(level=level, indexing_scheme=indexing_scheme)

        actual_lon, actual_lat = grid.cell_ids2geographic(cell_ids)

        np.testing.assert_allclose(actual_lon, expected[0])
        np.testing.assert_allclose(actual_lat, expected[1])

    @pytest.mark.parametrize(
        ["cell_centers", "level", "indexing_scheme", "expected"],
        (
            pytest.param(
                np.array([[315.0, 66.44353569089877]]),
                1,
                "ring",
                np.array([3]),
            ),
            pytest.param(
                np.array(
                    [[61.875, 19.47122063], [33.75, 24.62431835], [84.375, 41.8103149]]
                ),
                3,
                "nested",
                np.array([5, 11, 21]),
            ),
        ),
    )
    def test_geographic2cell_ids(
        self, cell_centers, level, indexing_scheme, expected
    ) -> None:
        grid = healpix.HealpixInfo(level=level, indexing_scheme=indexing_scheme)

        actual = grid.geographic2cell_ids(
            lon=cell_centers[:, 0], lat=cell_centers[:, 1]
        )

        np.testing.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        ["level", "cell_ids", "new_level", "expected"],
        (
            pytest.param(
                1,
                np.array([0, 4, 8, 12, 16]),
                0,
                np.array([0, 1, 2, 3, 4]),
                id="level1-parents",
            ),
            pytest.param(
                1,
                np.array([0, 1, 2, 3]),
                2,
                np.array(
                    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
                ),
                id="level1-children",
            ),
            pytest.param(
                1,
                np.array([0, 4]),
                3,
                np.stack([np.arange(16), 4 * 4**2 + np.arange(16)]),
                id="level1-grandchildren",
            ),
        ),
    )
    def test_zoom_to(self, level, cell_ids, new_level, expected):
        grid = healpix.HealpixInfo(level=level, indexing_scheme="nested")

        actual = grid.zoom_to(cell_ids, level=new_level)

        np.testing.assert_equal(actual, expected)

    def test_zoom_to_ring(self):
        cell_ids = np.array([1, 2, 3])
        grid = healpix.HealpixInfo(level=1, indexing_scheme="ring")

        with pytest.raises(ValueError, match="Scaling does not make sense.*'ring'.*"):
            grid.zoom_to(cell_ids, level=0)


@pytest.mark.parametrize(
    ["mapping", "expected"],
    (
        pytest.param(
            {"resolution": 10, "indexing_scheme": "nested"},
            {"level": 10, "indexing_scheme": "nested"},
            id="no_translation",
        ),
        pytest.param(
            {
                "level": 10,
                "indexing_scheme": "nested",
                "grid_name": "healpix",
            },
            {"level": 10, "indexing_scheme": "nested"},
            id="no_translation-grid_name",
        ),
        pytest.param(
            {"nside": 1024, "indexing_scheme": "nested"},
            {"level": 10, "indexing_scheme": "nested"},
            id="nside-alone",
        ),
        pytest.param(
            {
                "nside": 1024,
                "level": 10,
                "indexing_scheme": "nested",
            },
            ExceptionGroup(
                "received multiple values for parameters",
                [
                    ValueError(
                        "Parameter level received multiple values: ['level', 'nside']"
                    )
                ],
            ),
            id="nside-duplicated",
        ),
        pytest.param(
            {
                "level": 10,
                "indexing_scheme": "nested",
                "nest": True,
            },
            ExceptionGroup(
                "received multiple values for parameters",
                [
                    ValueError(
                        "Parameter indexing_scheme received multiple values: ['indexing_scheme', 'nest']"
                    ),
                ],
            ),
            id="indexing_scheme-duplicated",
        ),
        pytest.param(
            {
                "nside": 1024,
                "level": 10,
                "indexing_scheme": "nested",
                "nest": True,
            },
            ExceptionGroup(
                "received multiple values for parameters",
                [
                    ValueError(
                        "Parameter indexing_scheme received multiple values: ['indexing_scheme', 'nest']"
                    ),
                    ValueError(
                        "Parameter level received multiple values: ['level', 'nside']"
                    ),
                ],
            ),
            id="multiple_params-duplicated",
        ),
    ),
)
def test_healpix_info_from_dict(mapping, expected) -> None:
    if isinstance(expected, ExceptionGroup):
        with pytest.raises(type(expected), match=expected.args[0]) as actual:
            healpix.HealpixInfo.from_dict(mapping)
        assert_exceptions_equal(actual.value, expected)
        return
    actual = healpix.HealpixInfo.from_dict(mapping)
    assert actual == healpix.HealpixInfo(**expected)


class TestHealpixIndex:
    @given(strategies.cell_ids(), strategies.dims, strategies.grids())
    def test_init(self, cell_ids, dim, grid) -> None:
        index = healpix.HealpixIndex(cell_ids, dim, grid)

        assert index._grid == grid
        assert index._dim == dim
        assert index._index.dim == dim

        np.testing.assert_equal(index._index.index.values, cell_ids)

    @given(strategies.grids())
    def test_grid(self, grid):
        index = healpix.HealpixIndex([0], dim="cells", grid_info=grid)

        assert index.grid_info is grid


@pytest.mark.parametrize("options", options)
@pytest.mark.parametrize("variable", variables)
@pytest.mark.parametrize("variable_name", variable_names)
def test_from_variables(variable_name, variable, options) -> None:
    expected_level = variable.attrs["level"]
    expected_scheme = variable.attrs["indexing_scheme"]

    variables = {variable_name: variable}

    index = healpix.HealpixIndex.from_variables(variables, options=options)

    assert index._grid.level == expected_level
    assert index._grid.indexing_scheme == expected_scheme

    assert (index._dim,) == variable.dims
    np.testing.assert_equal(index._index.index.values, variable.data)


def test_from_variables_moc() -> None:
    level = 2
    grid_info = {"grid_name": "healpix", "level": level, "indexing_scheme": "nested"}
    variables = {"cell_ids": xr.Variable("cells", np.arange(12 * 4**level), grid_info)}

    index = healpix.HealpixIndex.from_variables(
        variables, options={"index_kind": "moc"}
    )

    assert isinstance(index._index, healpix.HealpixMocIndex)
    assert index.grid_info.to_dict() == grid_info


@pytest.mark.parametrize(["old_variable", "new_variable"], variable_combinations)
def test_replace(old_variable, new_variable) -> None:
    grid = healpix.HealpixInfo.from_dict(old_variable.attrs)

    index = healpix.HealpixIndex(
        cell_ids=old_variable.data,
        dim=old_variable.dims[0],
        grid_info=grid,
    )

    new_pandas_index = PandasIndex.from_variables(
        {"cell_ids": new_variable}, options={}
    )

    new_index = index._replace(new_pandas_index)

    assert new_index._dim == index._dim
    assert new_index._index == new_pandas_index
    assert index._grid == grid


@pytest.mark.parametrize("max_width", [20, 50, 80, 120])
@pytest.mark.parametrize("level", [0, 1, 3])
def test_repr_inline(level, max_width) -> None:
    grid_info = healpix.HealpixInfo(level=level, indexing_scheme="nested")
    index = healpix.HealpixIndex(cell_ids=[0], dim="cells", grid_info=grid_info)

    actual = index._repr_inline_(max_width)

    assert f"level={level}" in actual
    # ignore max_width for now
    # assert len(actual) <= max_width


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


def test_join():
    data1 = np.array([0, 5, 7, 9], dtype="uint64")
    data2 = np.array([0, 7])

    dim = "cells"
    grid_info = healpix.HealpixInfo(level=2)

    index1 = healpix.HealpixIndex(data1, dim=dim, grid_info=grid_info)
    index2 = healpix.HealpixIndex(data2, dim=dim, grid_info=grid_info)

    actual = index1.join(index2, how="inner")
    expected = healpix.HealpixIndex(data2, dim=dim, grid_info=grid_info)

    assert actual._grid == expected._grid
    assert actual._dim == expected._dim
    assert np.all(actual._index.index == expected._index.index)


def test_join_error():
    data1 = np.array([0, 7], dtype="uint64")
    data2 = np.array([5, 7, 9], dtype="uint64")

    dim = "cells"

    grid_info1 = healpix.HealpixInfo(level=1)
    grid_info2 = healpix.HealpixInfo(level=6)

    index1 = healpix.HealpixIndex(data1, dim=dim, grid_info=grid_info1)
    index2 = healpix.HealpixIndex(data2, dim=dim, grid_info=grid_info2)

    with pytest.raises(ValueError, match="different grid parameters"):
        index1.join(index2, how="inner")


def test_reindex_like():
    grid = healpix.HealpixInfo(level=2)
    index1 = healpix.HealpixIndex(
        cell_ids=np.array([0, 7]),
        dim="cells",
        grid_info=grid,
    )
    index2 = healpix.HealpixIndex(
        cell_ids=np.array([0, 5, 7, 9]),
        dim="cells",
        grid_info=grid,
    )

    actual = index1.reindex_like(index2)

    expected = {"cells": np.array([0, -1, 1, -1])}

    np.testing.assert_equal(actual["cells"], expected["cells"])


def test_reindex_like_error():
    data1 = np.array([0, 7], dtype="uint64")
    data2 = np.array([0, 5, 7], dtype="uint64")

    dim = "cells"

    grid_info1 = healpix.HealpixInfo(level=1)
    grid_info2 = healpix.HealpixInfo(level=6)

    index1 = healpix.HealpixIndex(data1, dim=dim, grid_info=grid_info1)
    index2 = healpix.HealpixIndex(data2, dim=dim, grid_info=grid_info2)

    with pytest.raises(ValueError, match="different grid parameters"):
        index1.reindex_like(index2)


@pytest.mark.parametrize(
    "variant", ("identical", "all-different", "dim", "grid-info", "values")
)
def test_equals(variant):
    values = [np.array([0, 7], dtype="uint64"), np.array([0, 5, 7], dtype="uint64")]
    dims = ["cells", "zones"]
    grid_info = [healpix.HealpixInfo(level=1), healpix.HealpixInfo(level=6)]

    dim1 = dims[0]
    values1 = values[0]
    grid_info1 = grid_info[0]

    variants = {
        "identical": (dims[0], values[0], grid_info[0]),
        "all-different": (dims[1], values[1], grid_info[1]),
        "dim": (dims[1], values[0], grid_info[0]),
        "grid-info": (dims[0], values[0], grid_info[1]),
        "values": (dims[0], values[1], grid_info[0]),
    }
    expected_results = {"identical": True}

    expected = expected_results.get(variant, False)
    dim2, values2, grid_info2 = variants[variant]

    index1 = healpix.HealpixIndex(values1, dim=dim1, grid_info=grid_info1)
    index2 = healpix.HealpixIndex(values2, dim=dim2, grid_info=grid_info2)

    assert index1.equals(index2) == expected
