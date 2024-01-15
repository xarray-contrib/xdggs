import itertools

import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from xarray.core.indexes import PandasIndex

from xdggs import healpix

resolutions = st.integers(min_value=0, max_value=60)
indexing_schemes = st.sampled_from(["nested", "ring", "unique"])

lon_rotation = st.floats(min_value=-180.0, max_value=360.0)
lat_rotation = st.floats(min_value=-90.0, max_value=90.0)
rotations = st.tuples(lon_rotation, lat_rotation)


class TestHealpixInfo:
    @given(resolutions, indexing_schemes, rotations)
    def test_init(self, resolution, indexing_scheme, rotation) -> None:
        grid = healpix.HealpixInfo(
            resolution=resolution, indexing_scheme=indexing_scheme, rotation=rotation
        )

        assert grid.resolution == resolution
        assert grid.indexing_scheme == indexing_scheme
        assert grid.rotation == rotation

    @given(resolutions)
    def test_nside(self, resolution):
        grid = healpix.HealpixInfo(resolution=resolution)

        assert grid.nside == 2**resolution

    @given(indexing_schemes)
    def test_nest(self, indexing_scheme):
        grid = healpix.HealpixInfo(resolution=1, indexing_scheme=indexing_scheme)
        if indexing_scheme not in {"nested", "ring"}:
            with pytest.raises(
                ValueError, match="cannot convert indexing scheme .* to `nest`"
            ):
                grid.nest
            return

        assert grid.nest == (True if indexing_scheme == "nested" else False)

    @given(resolutions, indexing_schemes, rotations)
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

    @given(resolutions, indexing_schemes, rotations)
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


cell_ids = [
    np.array([3]),
    np.array([5, 11, 21]),
    np.array([54, 70, 82, 91]),
]
cell_centers = [
    np.array([[45.0, 14.47751219]]),
    np.array([[61.875, 19.47122063], [33.75, 24.62431835], [84.375, 41.8103149]]),
    np.array(
        [
            [56.25, 66.44353569],
            [140.625, 19.47122063],
            [151.875, 30.0],
            [147.85714286, 48.14120779],
        ]
    ),
]

pixel_orderings = ["nested", "ring"]
resolutions = [1, 2, 8]
rotation = [(0, 0)]

options = [{}]
dims = ["cells", "zones"]
variable_names = ["cell_ids", "zonal_ids", "zone_ids"]
variables = [
    xr.Variable(
        dims[0],
        cell_ids[0],
        {
            "grid_type": "healpix",
            "nside": resolutions[0],
            "nest": True,
            "rotation": rotation[0],
        },
    ),
    xr.Variable(
        dims[1],
        cell_ids[0],
        {
            "grid_type": "healpix",
            "nside": resolutions[0],
            "nest": False,
            "rotation": rotation[0],
        },
    ),
    xr.Variable(
        dims[0],
        cell_ids[1],
        {
            "grid_type": "healpix",
            "nside": resolutions[1],
            "nest": True,
            "rotation": rotation[0],
        },
    ),
    xr.Variable(
        dims[1],
        cell_ids[2],
        {
            "grid_type": "healpix",
            "nside": resolutions[2],
            "nest": False,
            "rotation": rotation[0],
        },
    ),
]
variable_combinations = list(itertools.product(variables, repeat=2))


@pytest.mark.parametrize("rotation", rotation)
@pytest.mark.parametrize("pixel_ordering", pixel_orderings)
@pytest.mark.parametrize("resolution", resolutions)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("cell_ids", cell_ids)
def test_init(cell_ids, dim, resolution, pixel_ordering, rotation):
    index = healpix.HealpixIndex(
        cell_ids,
        dim,
        nside=resolution,
        nest=pixel_ordering == "nested",
        rot_latlon=rotation,
    )

    assert index._nside == resolution
    assert index._dim == dim

    assert index._pd_index.dim == dim
    np.testing.assert_equal(index._pd_index.index.values, cell_ids)


@pytest.mark.parametrize("options", options)
@pytest.mark.parametrize("variable", variables)
@pytest.mark.parametrize("variable_name", variable_names)
def test_from_variables(variable_name, variable, options):
    expected_resolution = variable.attrs["nside"]
    expected_scheme = variable.attrs["nest"]
    expected_rot = variable.attrs["rotation"]

    variables = {variable_name: variable}

    index = healpix.HealpixIndex.from_variables(variables, options=options)

    assert index._nside == expected_resolution
    assert index._nest == expected_scheme
    assert index._rot_latlon == expected_rot

    assert (index._dim,) == variable.dims
    np.testing.assert_equal(index._pd_index.index.values, variable.data)


@pytest.mark.parametrize(["old_variable", "new_variable"], variable_combinations)
def test_replace(old_variable, new_variable):
    index = healpix.HealpixIndex(
        cell_ids=old_variable.data,
        dim=old_variable.dims[0],
        nside=old_variable.attrs["nside"],
        nest=old_variable.attrs["nest"],
        rot_latlon=old_variable.attrs["rotation"],
    )

    new_pandas_index = PandasIndex.from_variables(
        {"cell_ids": new_variable}, options={}
    )

    new_index = index._replace(new_pandas_index)

    assert new_index._nside == index._nside
    assert new_index._nest == index._nest
    assert new_index._rot_latlon == index._rot_latlon
    assert new_index._dim == index._dim
    assert new_index._pd_index == new_pandas_index


@pytest.mark.parametrize(
    ["cell_ids", "cell_centers"], list(zip(cell_ids, cell_centers))
)
def test_cellid2latlon(cell_ids, cell_centers):
    index = healpix.HealpixIndex(
        cell_ids=[0], dim="cells", nside=8, nest=True, rot_latlon=(0, 0)
    )

    actual = index._cellid2latlon(cell_ids)
    expected = cell_centers

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ["cell_centers", "cell_ids"], list(zip(cell_centers, cell_ids))
)
def test_latlon2cell_ids(cell_centers, cell_ids):
    index = healpix.HealpixIndex(
        cell_ids=[0], dim="cells", nside=8, nest=True, rot_latlon=(0, 0)
    )

    actual = index._latlon2cellid(lon=cell_centers[:, 0], lat=cell_centers[:, 1])
    expected = cell_ids

    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("max_width", [20, 50, 80, 120])
@pytest.mark.parametrize("resolution", resolutions)
def test_repr_inline(resolution, max_width):
    index = healpix.HealpixIndex(
        cell_ids=[0], dim="cells", nside=resolution, nest=True, rot_latlon=(0, 0)
    )

    actual = index._repr_inline_(max_width)

    assert f"nside={resolution}" in actual
    # ignore max_width for now
    # assert len(actual) <= max_width
