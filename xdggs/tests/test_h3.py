import numpy as np
import pytest
import xarray as xr

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
        dims[1], cell_ids[1], {"grid_type": "h3", "resolution": resolutions[2]}
    ),
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
    variables = {variable_name: variable}
    index = h3.H3Index.from_variables(variables, options=options)

    assert index._resolution == variable.attrs["resolution"]
    assert (index._dim,) == variable.dims

    # TODO: how do we check the index, if at all?
    assert (index._pd_index.dim,) == variable.dims
    assert np.all(index._pd_index.index.values == variable.data)
