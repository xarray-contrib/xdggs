import numpy as np
import pytest
import xarray as xr
from arro3.core import Array, Table

from xdggs import plotting


@pytest.mark.parametrize(
    ["polygons", "arr", "coords", "expected"],
    (
        pytest.param(
            Array.from_numpy(np.array([1, 2])),
            xr.DataArray(
                [-1, 1],
                coords={
                    "cell_ids": ("cells", [0, 1]),
                    "latitude": ("cells", [-5, 10]),
                    "longitude": ("cells", [-60, -50]),
                },
                dims="cells",
            ),
            None,
            Table.from_pydict(
                {
                    "geometry": Array.from_numpy(np.array([1, 2])),
                    "cell_ids": Array.from_numpy(np.array([0, 1])),
                    "data": Array.from_numpy(np.array([-1, 1])),
                    "latitude": Array.from_numpy(np.array([-5, 10])),
                    "longitude": Array.from_numpy(np.array([-60, -50])),
                }
            ),
        ),
        pytest.param(
            Array.from_numpy(np.array([1, 2])),
            xr.DataArray(
                [-1, 1],
                coords={
                    "cell_ids": ("cells", [1, 2]),
                    "latitude": ("cells", [-5, 10]),
                    "longitude": ("cells", [-60, -50]),
                },
                dims="cells",
            ),
            ["latitude"],
            Table.from_pydict(
                {
                    "geometry": Array.from_numpy(np.array([1, 2])),
                    "cell_ids": Array.from_numpy(np.array([1, 2])),
                    "data": Array.from_numpy(np.array([-1, 1])),
                    "latitude": Array.from_numpy(np.array([-5, 10])),
                }
            ),
        ),
        pytest.param(
            Array.from_numpy(np.array([1, 3])),
            xr.DataArray(
                [-1, 1],
                coords={
                    "cell_ids": ("cells", [0, 1]),
                    "latitude": ("cells", [-5, 10]),
                    "longitude": ("cells", [-60, -50]),
                },
                dims="cells",
                name="new_data",
            ),
            ["longitude"],
            Table.from_pydict(
                {
                    "geometry": Array.from_numpy(np.array([1, 3])),
                    "cell_ids": Array.from_numpy(np.array([0, 1])),
                    "new_data": Array.from_numpy(np.array([-1, 1])),
                    "longitude": Array.from_numpy(np.array([-60, -50])),
                }
            ),
        ),
    ),
)
def test_create_arrow_table(polygons, arr, coords, expected):
    actual = plotting.create_arrow_table(polygons, arr, coords=coords)

    assert actual == expected


@pytest.mark.parametrize(
    ["var", "center", "expected"],
    (
        pytest.param(
            xr.Variable("cells", np.array([-5, np.nan, -2, 1])),
            None,
            np.array([0, np.nan, 0.5, 1]),
            id="linear-missing_values",
        ),
        pytest.param(
            xr.Variable("cells", np.arange(-5, 2, dtype="float")),
            None,
            np.linspace(0, 1, 7),
            id="linear-manual",
        ),
        pytest.param(
            xr.Variable("cells", np.linspace(0, 10, 5)),
            None,
            np.linspace(0, 1, 5),
            id="linear-linspace",
        ),
        pytest.param(
            xr.Variable("cells", np.linspace(-5, 5, 10)),
            0,
            np.linspace(0, 1, 10),
            id="centered-0",
        ),
        pytest.param(
            xr.Variable("cells", np.linspace(0, 10, 10)),
            5,
            np.linspace(0, 1, 10),
            id="centered-2",
        ),
    ),
)
def test_normalize(var, center, expected):
    actual = plotting.normalize(var, center=center)

    np.testing.assert_allclose(actual, expected)
