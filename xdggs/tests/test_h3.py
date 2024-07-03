import itertools

import numpy as np
import pytest
import xarray as xr
from xarray.core.indexes import PandasIndex

from xdggs import h3

# from the h3 gallery, at resolution 3
cell_ids = [
    np.array([0x832830FFFFFFFFF]),
    np.array([0x832831FFFFFFFFF, 0x832832FFFFFFFFF]),
    np.array([0x832833FFFFFFFFF, 0x832834FFFFFFFFF, 0x832835FFFFFFFFF]),
]
cell_centers = [
    np.array([[-122.19619676, 38.19320895]]),
    np.array([[-123.43390346, 38.63853196], [-121.00991811, 38.82387033]]),
    np.array(
        [
            [-122.2594399, 39.27846774],
            [-122.13425086, 37.09786649],
            [-123.35925909, 37.55231005],
        ]
    ),
]
dims = ["cells", "zones"]
resolutions = [1, 5, 15]
variable_names = ["cell_ids", "zonal_ids", "zone_ids"]

variables = [
    xr.Variable(
        dims[0], cell_ids[0], {"grid_name": "h3", "resolution": resolutions[0]}
    ),
    xr.Variable(
        dims[1], cell_ids[0], {"grid_name": "h3", "resolution": resolutions[0]}
    ),
    xr.Variable(
        dims[0], cell_ids[1], {"grid_name": "h3", "resolution": resolutions[1]}
    ),
    xr.Variable(
        dims[1], cell_ids[2], {"grid_name": "h3", "resolution": resolutions[2]}
    ),
]
variable_combinations = [
    (old, new) for old, new in itertools.product(variables, repeat=2)
]


class TestH3Info:

    @pytest.mark.parametrize(
        ["resolution", "error"],
        (
            (0, None),
            (1, None),
            (-1, ValueError("resolution must be an integer between")),
        ),
    )
    def test_init(self, resolution, error):
        if error is not None:
            with pytest.raises(type(error), match=str(error)):
                h3.H3Info(resolution=resolution)
            return

        actual = h3.H3Info(resolution=resolution)

        assert actual.resolution == resolution

    @pytest.mark.parametrize(
        ["mapping", "expected"],
        (
            ({"resolution": 0}, 0),
            ({"resolution": 1}, 1),
            ({"resolution": -1}, ValueError("resolution must be an integer between")),
        ),
    )
    def test_from_dict(self, mapping, expected):
        if isinstance(expected, Exception):
            with pytest.raises(type(expected), match=str(expected)):
                h3.H3Info.from_dict(mapping)
            return

        actual = h3.H3Info.from_dict(mapping)
        assert actual.resolution == expected

    def test_roundtrip(self):
        mapping = {"grid_name": "h3", "resolution": 0}

        grid = h3.H3Info.from_dict(mapping)
        actual = grid.to_dict()

        assert actual == mapping

    @pytest.mark.parametrize(
        ["cell_ids", "cell_centers"], list(zip(cell_ids, cell_centers))
    )
    def test_cell_ids2geographic(self, cell_ids, cell_centers):
        grid = h3.H3Info(resolution=3)

        actual = grid.cell_ids2geographic(cell_ids)
        expected = cell_centers.T

        assert isinstance(actual, tuple) and len(actual) == 2
        np.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        ["cell_centers", "cell_ids"], list(zip(cell_centers, cell_ids))
    )
    def test_geographic2cell_ids(self, cell_centers, cell_ids):
        grid = h3.H3Info(resolution=3)

        actual = grid.geographic2cell_ids(
            lon=cell_centers[:, 0], lat=cell_centers[:, 1]
        )
        expected = cell_ids

        np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("resolution", resolutions)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("cell_ids", cell_ids)
def test_init(cell_ids, dim, resolution):
    grid = h3.H3Info(resolution)
    index = h3.H3Index(cell_ids, dim, grid)

    assert index._grid == grid
    assert index._dim == dim

    # TODO: how do we check the index, if at all?
    assert index._pd_index.dim == dim
    assert np.all(index._pd_index.index.values == cell_ids)


@pytest.mark.parametrize("resolution", resolutions)
def test_grid(resolution):
    grid = h3.H3Info(resolution)

    index = h3.H3Index([0], "cell_ids", grid)

    assert index.grid_info is grid


@pytest.mark.parametrize("variable", variables)
@pytest.mark.parametrize("variable_name", variable_names)
@pytest.mark.parametrize("options", [{}])
def test_from_variables(variable_name, variable, options):
    expected_resolution = variable.attrs["resolution"]

    variables = {variable_name: variable}
    index = h3.H3Index.from_variables(variables, options=options)

    assert index._grid.resolution == expected_resolution
    assert (index._dim,) == variable.dims

    # TODO: how do we check the index, if at all?
    assert (index._pd_index.dim,) == variable.dims
    assert np.all(index._pd_index.index.values == variable.data)


@pytest.mark.parametrize(["old_variable", "new_variable"], variable_combinations)
def test_replace(old_variable, new_variable):
    grid = h3.H3Info(resolution=old_variable.attrs["resolution"])
    index = h3.H3Index(
        cell_ids=old_variable.data,
        dim=old_variable.dims[0],
        grid_info=grid,
    )
    new_pandas_index = PandasIndex.from_variables(
        {"cell_ids": new_variable}, options={}
    )

    new_index = index._replace(new_pandas_index)

    assert new_index._grid == index._grid
    assert new_index._dim == index._dim
    assert new_index._pd_index == new_pandas_index


@pytest.mark.parametrize("max_width", [20, 50, 80, 120])
@pytest.mark.parametrize("resolution", resolutions)
def test_repr_inline(resolution, max_width):
    grid = h3.H3Info(resolution=resolution)
    index = h3.H3Index(cell_ids=[0], dim="cells", grid_info=grid)

    actual = index._repr_inline_(max_width)

    assert f"resolution={resolution}" in actual
    # ignore max_width for now
    # assert len(actual) <= max_width
