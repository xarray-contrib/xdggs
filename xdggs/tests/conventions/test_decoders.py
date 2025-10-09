import numpy as np
import pytest
import xarray as xr

import xdggs
from xdggs.conventions import decoders


def clear_attrs(var):
    new_var = var.copy(deep=False)
    new_var.attrs.clear()

    return new_var


def generate_h3_cells(level):
    from h3ronpy import change_resolution

    base_cells = [
        576495936675512319,
        576531121047601151,
        576566305419689983,
        576601489791778815,
        576636674163867647,
        576671858535956479,
        576707042908045311,
        576742227280134143,
        576777411652222975,
        576812596024311807,
    ]

    return change_resolution(base_cells, level)


def create_coordinate(grid_name, dim, level, **options):
    generators = {
        "healpix": lambda level: np.arange(12 * 4**level),
        "h3": generate_h3_cells,
    }

    cell_ids = generators[grid_name](level)
    grid_info = {"grid_name": grid_name, "refinement_level": level, **options}
    return xr.Variable(dim, cell_ids, grid_info)


def create_index(name, coord, grid_info):
    translations = {"refinement_level": "level"}

    if grid_info is None:
        grid_info = {
            translations.get(name, name): value for name, value in coord.attrs.items()
        }

    var = coord.copy(deep=False)
    var.attrs = grid_info

    return xdggs.HealpixIndex.from_variables({name: var}, options={})


@pytest.mark.parametrize("obj_type", ["DataArray", "Dataset"])
@pytest.mark.parametrize(
    ["coord_name", "coord", "grid_info", "name"],
    (
        pytest.param(
            "cell_ids",
            create_coordinate("healpix", "cells", 0, indexing_scheme="ring"),
            None,
            None,
            id="healpix-all_defaults",
        ),
        pytest.param(
            "cell_ids",
            create_coordinate("h3", "cells", 0),
            None,
            None,
            id="h3-all_defaults",
        ),
        pytest.param(
            "zone_ids",
            create_coordinate("healpix", "zones", 0, indexing_scheme="nested"),
            None,
            "zone_ids",
            id="healpix-override_name",
        ),
        pytest.param(
            "cell_ids",
            clear_attrs(create_coordinate("h3", "cells", 2)),
            {"grid_name": "h3", "level": 2},
            None,
            id="h3-override_grid_info",
        ),
    ),
)
def test_xdggs(obj_type, coord_name, coord, grid_info, name):
    if obj_type == "Dataset":
        obj = xr.Dataset(coords={coord_name: coord})
    else:
        obj = xr.DataArray(
            np.zeros_like(coord.data), coords={coord_name: coord}, dims=coord.dims
        )

    expected = xr.Coordinates(
        {coord_name: coord},
        indexes={coord_name: create_index(coord_name, coord, grid_info)},
    )
    actual = decoders.xdggs(obj, grid_info, name)
    xr.testing.assert_identical(actual.to_dataset(), expected.to_dataset())
