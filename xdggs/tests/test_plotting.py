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
                    "latitude": ("cells", [-5, 10]),
                    "longitude": ("cells", [-60, -50]),
                },
                dims="cells",
            ),
            None,
            Table.from_pydict(
                {
                    "geometry": Array.from_numpy(np.array([1, 2])),
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
                    "latitude": ("cells", [-5, 10]),
                    "longitude": ("cells", [-60, -50]),
                },
                dims="cells",
            ),
            ["latitude"],
            Table.from_pydict(
                {
                    "geometry": Array.from_numpy(np.array([1, 2])),
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
