import pytest
import xarray as xr

import xdggs.utils


@pytest.mark.parametrize("name", ["h3", "s2", "rhealpix", "healpix"])
def test_register_dggs(monkeypatch, name):
    registry = {}

    monkeypatch.setattr(xdggs.utils, "GRID_REGISTRY", registry)

    grid = object()
    registered_grid = xdggs.utils.register_dggs(name)(grid)

    assert grid is registered_grid

    assert len(registry) == 1
    assert name in registry and registry[name] is grid


@pytest.mark.parametrize(
    ["variables", "expected"],
    (
        (
            {"cell_ids": xr.Variable("cells", [0, 1])},
            ("cell_ids", "cells"),
        ),
        (
            {"zone_ids": xr.Variable("zones", [0, 1])},
            ("zone_ids", "zones"),
        ),
    ),
)
def test_extract_cell_id_variable(variables, expected):
    expected_var = variables[expected[0]]

    actual = xdggs.utils._extract_cell_id_variable(variables)
    actual_var = actual[1]

    assert actual[::2] == expected
    xr.testing.assert_equal(actual_var, expected_var)
