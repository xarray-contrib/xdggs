import itertools

import numpy as np
import pytest
import xarray as xr
from xarray.core.indexes import PandasIndex

from xdggs import healpix

cell_ids = [
    np.array([3]),
    np.array([5, 11, 21]),
    np.array([54, 70, 82, 91]),
]
pixel_orderings = ["nested", "ring"]
resolutions = [0, 1, 3]
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
