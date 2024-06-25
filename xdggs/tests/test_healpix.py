import itertools

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
import xarray.testing.strategies as xrst
from hypothesis import given
from xarray.core.indexes import PandasIndex

from xdggs import healpix
from xdggs.tests import assert_exceptions_equal

try:
    ExceptionGroup
except NameError:  # pragma: no cover
    from exceptiongroup import ExceptionGroup


# namespace class
class strategies:
    invalid_resolutions = st.integers(max_value=-1) | st.integers(min_value=30)
    resolutions = st.integers(min_value=0, max_value=29)
    indexing_schemes = st.sampled_from(["nested", "ring", "unique"])
    invalid_indexing_schemes = st.text().filter(
        lambda x: x not in ["nested", "ring", "unique"]
    )

    def rotations():
        lon_rotation = st.floats(min_value=-180.0, max_value=360.0)
        lat_rotation = st.floats(min_value=-90.0, max_value=90.0)
        return st.tuples(lon_rotation, lat_rotation)

    dims = xrst.names()

    @classmethod
    def grid_mappings(cls):
        strategies = {
            "resolution": cls.resolutions,
            "nside": cls.resolutions.map(lambda n: 2**n),
            "depth": cls.resolutions,
            "level": cls.resolutions,
            "order": cls.resolutions,
            "indexing_scheme": cls.indexing_schemes,
            "nest": st.booleans(),
            "rotation": cls.rotations(),
            "rot_latlon": cls.rotations().map(lambda x: x[::-1]),
        }

        names = {
            "resolution": st.sampled_from(
                ["resolution", "nside", "depth", "level", "order"]
            ),
            "indexing_scheme": st.sampled_from(["indexing_scheme", "nest"]),
            "rotation": st.sampled_from(["rotation", "rot_latlon"]),
        }

        return st.builds(lambda **x: list(x.values()), **names).flatmap(
            lambda params: st.builds(dict, **{p: strategies[p] for p in params})
        )

    def cell_ids(max_value=None, dtypes=None):
        if dtypes is None:
            # healpy can't deal with `uint32` or less (it segfaults occasionally)
            dtypes = st.sampled_from(["uint64"])
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
        resolutions=resolutions,
        indexing_schemes=indexing_schemes,
        rotations=rotations(),
    ):
        return st.builds(
            healpix.HealpixInfo,
            resolution=resolutions,
            indexing_scheme=indexing_schemes,
            rotation=rotations,
        )

    @classmethod
    def grid_and_cell_ids(
        cls,
        resolutions=resolutions,
        indexing_schemes=indexing_schemes,
        rotations=rotations(),
        dtypes=None,
    ):
        cell_resolutions = st.shared(resolutions, key="common-resolutions")
        grid_resolutions = st.shared(resolutions, key="common-resolutions")
        cell_ids_ = cell_resolutions.flatmap(
            lambda resolution: cls.cell_ids(
                max_value=12 * 2 ** (resolution * 2) - 1, dtypes=dtypes
            )
        )
        grids_ = cls.grids(
            resolutions=grid_resolutions,
            indexing_schemes=indexing_schemes,
            rotations=rotations,
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
            "resolution": 0,
            "indexing_scheme": "nested",
            "rotation": (0, 0),
        },
    ),
    xr.Variable(
        "zones",
        np.array([3]),
        {
            "grid_name": "healpix",
            "resolution": 0,
            "indexing_scheme": "ring",
            "rotation": (0, 0),
        },
    ),
    xr.Variable(
        "cells",
        np.array([5, 11, 21]),
        {
            "grid_name": "healpix",
            "resolution": 1,
            "indexing_scheme": "nested",
            "rotation": (0, 0),
        },
    ),
    xr.Variable(
        "zones",
        np.array([54, 70, 82, 91]),
        {
            "grid_name": "healpix",
            "resolution": 3,
            "indexing_scheme": "nested",
            "rotation": (0, 0),
        },
    ),
]
variable_combinations = list(itertools.product(variables, repeat=2))


class TestHealpixInfo:
    @given(strategies.invalid_resolutions)
    def test_init_invalid_resolutions(self, resolution):
        with pytest.raises(
            ValueError, match="resolution must be an integer in the range of"
        ):
            healpix.HealpixInfo(resolution=resolution)

    @given(strategies.invalid_indexing_schemes)
    def test_init_invalid_indexing_scheme(self, indexing_scheme):
        with pytest.raises(ValueError, match="indexing scheme must be one of"):
            healpix.HealpixInfo(
                resolution=0,
                indexing_scheme=indexing_scheme,
            )

    @given(strategies.resolutions, strategies.indexing_schemes, strategies.rotations())
    def test_init(self, resolution, indexing_scheme, rotation):
        grid = healpix.HealpixInfo(
            resolution=resolution, indexing_scheme=indexing_scheme, rotation=rotation
        )

        assert grid.resolution == resolution
        assert grid.indexing_scheme == indexing_scheme
        assert grid.rotation == rotation

    @given(strategies.resolutions)
    def test_nside(self, resolution):
        grid = healpix.HealpixInfo(resolution=resolution)

        assert grid.nside == 2**resolution

    @given(strategies.indexing_schemes)
    def test_nest(self, indexing_scheme):
        grid = healpix.HealpixInfo(resolution=1, indexing_scheme=indexing_scheme)
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

    @given(strategies.resolutions, strategies.indexing_schemes, strategies.rotations())
    def test_to_dict(self, resolution, indexing_scheme, rotation) -> None:
        grid = healpix.HealpixInfo(
            resolution=resolution, indexing_scheme=indexing_scheme, rotation=rotation
        )
        actual = grid.to_dict()

        assert set(actual) == {"grid_name", "resolution", "indexing_scheme", "rotation"}
        assert actual["grid_name"] == "healpix"
        assert actual["resolution"] == resolution
        assert actual["indexing_scheme"] == indexing_scheme
        assert actual["rotation"] == rotation

    @given(strategies.resolutions, strategies.indexing_schemes, strategies.rotations())
    def test_roundtrip(self, resolution, indexing_scheme, rotation):
        mapping = {
            "grid_name": "healpix",
            "resolution": resolution,
            "indexing_scheme": indexing_scheme,
            "rotation": rotation,
        }

        grid = healpix.HealpixInfo.from_dict(mapping)
        roundtripped = grid.to_dict()

        assert roundtripped == mapping

    @given(
        *strategies.grid_and_cell_ids(
            indexing_schemes=st.sampled_from(["nested", "ring"]),
            dtypes=st.sampled_from(["int64"]),
        )
    )
    def test_cell_center_roundtrip(self, cell_ids, grid) -> None:
        centers = grid.cell_ids2geographic(cell_ids)

        roundtripped = grid.geographic2cell_ids(lat=centers[1], lon=centers[0])

        np.testing.assert_equal(roundtripped, cell_ids)

    @pytest.mark.parametrize(
        ["cell_ids", "resolution", "indexing_scheme", "expected"],
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
        self, cell_ids, resolution, indexing_scheme, expected
    ) -> None:
        grid = healpix.HealpixInfo(
            resolution=resolution, indexing_scheme=indexing_scheme
        )

        actual_lon, actual_lat = grid.cell_ids2geographic(cell_ids)

        np.testing.assert_allclose(actual_lon, expected[0])
        np.testing.assert_allclose(actual_lat, expected[1])

    @pytest.mark.parametrize(
        ["cell_centers", "resolution", "indexing_scheme", "expected"],
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
        self, cell_centers, resolution, indexing_scheme, expected
    ) -> None:
        grid = healpix.HealpixInfo(
            resolution=resolution, indexing_scheme=indexing_scheme
        )

        actual = grid.geographic2cell_ids(
            lon=cell_centers[:, 0], lat=cell_centers[:, 1]
        )

        np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(
    ["mapping", "expected"],
    (
        pytest.param(
            {"resolution": 10, "indexing_scheme": "nested", "rotation": (0.0, 0.0)},
            {"resolution": 10, "indexing_scheme": "nested", "rotation": (0.0, 0.0)},
            id="no_translation",
        ),
        pytest.param(
            {
                "resolution": 10,
                "indexing_scheme": "nested",
                "rotation": (0.0, 0.0),
                "grid_name": "healpix",
            },
            {"resolution": 10, "indexing_scheme": "nested", "rotation": (0.0, 0.0)},
            id="no_translation-grid_name",
        ),
        pytest.param(
            {"nside": 1024, "indexing_scheme": "nested", "rotation": (0.0, 0.0)},
            {"resolution": 10, "indexing_scheme": "nested", "rotation": (0.0, 0.0)},
            id="nside-alone",
        ),
        pytest.param(
            {
                "nside": 1024,
                "resolution": 10,
                "indexing_scheme": "nested",
                "rotation": (0.0, 0.0),
            },
            ExceptionGroup(
                "received multiple values for parameters",
                [
                    ValueError(
                        "Parameter resolution received multiple values: ['nside', 'resolution']"
                    )
                ],
            ),
            id="nside-duplicated",
        ),
        pytest.param(
            {
                "resolution": 10,
                "indexing_scheme": "nested",
                "nest": True,
                "rotation": (0.0, 0.0),
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
                "resolution": 10,
                "indexing_scheme": "nested",
                "nest": True,
                "rotation": (0.0, 0.0),
            },
            ExceptionGroup(
                "received multiple values for parameters",
                [
                    ValueError(
                        "Parameter indexing_scheme received multiple values: ['indexing_scheme', 'nest']"
                    ),
                    ValueError(
                        "Parameter resolution received multiple values: ['nside', 'resolution']"
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
        assert index._pd_index.dim == dim

        np.testing.assert_equal(index._pd_index.index.values, cell_ids)

    @given(strategies.grids())
    def test_grid(self, grid):
        index = healpix.HealpixIndex([0], dim="cells", grid_info=grid)

        assert index.grid_info is grid


@pytest.mark.parametrize("options", options)
@pytest.mark.parametrize("variable", variables)
@pytest.mark.parametrize("variable_name", variable_names)
def test_from_variables(variable_name, variable, options) -> None:
    expected_resolution = variable.attrs["resolution"]
    expected_scheme = variable.attrs["indexing_scheme"]
    expected_rot = variable.attrs["rotation"]

    variables = {variable_name: variable}

    index = healpix.HealpixIndex.from_variables(variables, options=options)

    assert index._grid.resolution == expected_resolution
    assert index._grid.indexing_scheme == expected_scheme
    assert index._grid.rotation == expected_rot

    assert (index._dim,) == variable.dims
    np.testing.assert_equal(index._pd_index.index.values, variable.data)


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
    assert new_index._pd_index == new_pandas_index
    assert index._grid == grid


@pytest.mark.parametrize("max_width", [20, 50, 80, 120])
@pytest.mark.parametrize("resolution", [0, 1, 3])
def test_repr_inline(resolution, max_width) -> None:
    grid_info = healpix.HealpixInfo(
        resolution=resolution, indexing_scheme="nested", rotation=(0, 0)
    )
    index = healpix.HealpixIndex(cell_ids=[0], dim="cells", grid_info=grid_info)

    actual = index._repr_inline_(max_width)

    assert f"nside={resolution}" in actual
    # ignore max_width for now
    # assert len(actual) <= max_width
