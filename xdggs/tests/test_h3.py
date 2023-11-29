import itertools

import numpy as np
import pytest
import xarray as xr
from xarray.core.indexes import PandasIndex

from xdggs import h3

cell_ids = [
    np.array([0x832830FFFFFFFFF]),
    np.array([0x832831FFFFFFFFF, 0x832832FFFFFFFFF]),
    np.array([0x832833FFFFFFFFF, 0x832834FFFFFFFFF, 0x832835FFFFFFFFF]),
]
dims = ["cells", "zones"]
resolutions = [1, 5, 15]
variable_names = ["cell_ids", "zonal_ids", "zone_ids"]

variables = [
    xr.Variable(
        dims[0], cell_ids[0], {"grid_type": "h3", "resolution": resolutions[0]}
    ),
    xr.Variable(
        dims[1], cell_ids[0], {"grid_type": "h3", "resolution": resolutions[0]}
    ),
    xr.Variable(
        dims[0], cell_ids[1], {"grid_type": "h3", "resolution": resolutions[1]}
    ),
    xr.Variable(
        dims[1], cell_ids[2], {"grid_type": "h3", "resolution": resolutions[2]}
    ),
]
variable_combinations = [
    (old, new) for old, new in itertools.product(variables, repeat=2)
]


@pytest.mark.parametrize("resolution", resolutions)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("cell_ids", cell_ids)
def test_init(cell_ids, dim, resolution):
    index = h3.H3Index(cell_ids, dim, resolution)

    assert index._resolution == resolution
    assert index._dim == dim

    # TODO: how do we check the index, if at all?
    assert index._pd_index.dim == dim
    assert np.all(index._pd_index.index.values == cell_ids)


@pytest.mark.parametrize("variable", variables)
@pytest.mark.parametrize("variable_name", variable_names)
@pytest.mark.parametrize("options", [{}])
def test_from_variables(variable_name, variable, options):
    expected_resolution = variable.attrs["resolution"]

    variables = {variable_name: variable}
    index = h3.H3Index.from_variables(variables, options=options)

    assert index._resolution == expected_resolution
    assert (index._dim,) == variable.dims

    # TODO: how do we check the index, if at all?
    assert (index._pd_index.dim,) == variable.dims
    assert np.all(index._pd_index.index.values == variable.data)


@pytest.mark.parametrize(["old_variable", "new_variable"], variable_combinations)
def test_replace(old_variable, new_variable):
    index = h3.H3Index(
        cell_ids=old_variable.data,
        dim=old_variable.dims[0],
        resolution=old_variable.attrs["resolution"],
    )
    new_pandas_index = PandasIndex.from_variables(
        {"cell_ids": new_variable}, options={}
    )

    new_index = index._replace(new_pandas_index)

    assert new_index._resolution == index._resolution
    assert new_index._dim == index._dim
    assert new_index._pd_index == new_pandas_index
