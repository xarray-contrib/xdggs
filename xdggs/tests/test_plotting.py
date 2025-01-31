import ipywidgets
import lonboard
import numpy as np
import pytest
import xarray as xr
from arro3.core import Array, Table
from matplotlib import colormaps

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


@pytest.mark.parametrize(
    ["var", "kwargs", "expected"],
    (
        pytest.param(
            xr.Variable("cells", [0, 3]),
            {"center": 2, "colormap": colormaps["viridis"], "alpha": 1},
            np.array([[68, 1, 84], [94, 201, 97]], dtype="uint8"),
        ),
        pytest.param(
            xr.Variable("cells", [-1, 1]),
            {"center": None, "colormap": colormaps["viridis"], "alpha": 0.8},
            np.array([[68, 1, 84, 204], [253, 231, 36, 204]], dtype="uint8"),
        ),
    ),
)
def test_colorize(var, kwargs, expected):
    actual = plotting.colorize(var, **kwargs)

    np.testing.assert_equal(actual, expected)


class TestMapContainer:
    def test_init(self):
        map_ = lonboard.Map(layers=[])
        sliders = ipywidgets.VBox(
            [ipywidgets.IntSlider(min=0, max=10, description="time")]
        )
        obj = xr.DataArray([[0, 1], [2, 3]], dims=["time", "cells"])
        colorize_kwargs = {"a": 1, "b": 2}

        container = plotting.MapContainer(
            dimension_sliders=sliders,
            map=map_,
            obj=obj,
            colorize_kwargs=colorize_kwargs,
        )

        assert container.map == map_
        xr.testing.assert_equal(container.obj, obj)
        assert container.dimension_sliders == sliders
        assert container.colorize_kwargs == colorize_kwargs

    def test_render(self):
        map_ = lonboard.Map(layers=[])
        sliders = ipywidgets.VBox(
            [ipywidgets.IntSlider(min=0, max=10, description="time")]
        )
        obj = xr.DataArray([[0, 1], [2, 3]], dims=["time", "cells"])
        colorize_kwargs = {"a": 1, "b": 2}

        container = plotting.MapContainer(
            dimension_sliders=sliders,
            map=map_,
            obj=obj,
            colorize_kwargs=colorize_kwargs,
        )
        rendered = container.render()

        assert isinstance(rendered, ipywidgets.VBox)


@pytest.mark.parametrize(
    ["arr", "expected_type"],
    (
        pytest.param(
            xr.DataArray(
                [0, 1], coords={"cell_ids": ("cells", [10, 26])}, dims="cells"
            ).dggs.decode(
                {"grid_name": "healpix", "level": 1, "indexing_scheme": "nested"}
            ),
            lonboard.Map,
            id="1d",
        ),
        pytest.param(
            xr.DataArray(
                [[0, 1], [2, 3]],
                coords={"cell_ids": ("cells", [10, 26])},
                dims=["time", "cells"],
            ).dggs.decode(
                {"grid_name": "healpix", "level": 1, "indexing_scheme": "nested"}
            ),
            ipywidgets.VBox,
            id="2d",
        ),
    ),
)
def test_explore(arr, expected_type):
    actual = arr.dggs.explore()

    assert isinstance(actual, expected_type)
