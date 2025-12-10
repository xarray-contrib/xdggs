import ipywidgets
import lonboard
import numpy as np
import pytest
import xarray as xr
from arro3.core import Array, Table
from matplotlib import colormaps

from xdggs import plotting
from xdggs.plotting.colorize import ColorizeParameters


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
        pytest.param(
            Array.from_numpy(np.array([1, 3])),
            xr.DataArray(
                np.arange(4).reshape((2, 2))[:, 0],
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
                    "new_data": Array.from_numpy(np.array([0, 2])),
                    "longitude": Array.from_numpy(np.array([-60, -50])),
                }
            ),
            id="non-contiguous",
        ),
    ),
)
def test_create_arrow_table(polygons, arr, coords, expected):
    actual = plotting.arrow.create_arrow_table(polygons, arr, coords=coords)

    assert actual == expected


@pytest.mark.parametrize(
    ["var", "params", "expected_values", "expected_stats"],
    (
        pytest.param(
            xr.Variable("cells", np.array([-5, np.nan, -2, 1])),
            ColorizeParameters(),
            np.array([0, np.nan, 0.5, 1]),
            {"vmin": -5.0, "vmax": 1.0},
            id="linear-missing_values",
        ),
        pytest.param(
            xr.Variable("cells", np.arange(-5, 2, dtype="float")),
            ColorizeParameters(),
            np.linspace(0, 1, 7),
            {"vmin": -5.0, "vmax": 1.0},
            id="linear-manual",
        ),
        pytest.param(
            xr.Variable("cells", np.linspace(0, 10, 5)),
            ColorizeParameters(),
            np.linspace(0, 1, 5),
            {"vmin": 0.0, "vmax": 10.0},
            id="linear-linspace",
        ),
        pytest.param(
            xr.Variable("cells", np.linspace(-5, 5, 10)),
            ColorizeParameters(center=0),
            np.linspace(0, 1, 10),
            {"vmin": -5.0, "vmax": 5.0},
            id="centered-0",
        ),
        pytest.param(
            xr.Variable("cells", np.linspace(0, 10, 10)),
            ColorizeParameters(center=5),
            np.linspace(0, 1, 10),
            {"vmin": 0.0, "vmax": 10.0},
            id="centered-2",
        ),
        pytest.param(
            xr.Variable("cells", np.arange(5, dtype="float64")),
            ColorizeParameters(vmin=1, vmax=3),
            np.array([-0.5, 0, 0.5, 1, 1.5], dtype="float64"),
            {"vmin": 1.0, "vmax": 3.0},
            id="vmin-vmax",
        ),
        pytest.param(
            xr.Variable("cells", np.linspace(0, 1, 3)),
            ColorizeParameters(robust=True),
            np.array([-0.020833333, 0.5, 1.020833333], dtype="float64"),
            {"vmin": 0.02, "vmax": 0.98},
            id="robust",
        ),
        pytest.param(
            xr.Variable("cells", np.linspace(-1.5, 1, 4)),
            ColorizeParameters(center=0.0, robust=True),
            np.array(
                [-0.010204082, 0.27324263, 0.556689342, 0.840136054], dtype="float64"
            ),
            {"vmin": -1.47, "vmax": 1.47},
            id="centered-robust",
        ),
    ),
)
def test_normalize(var, params, expected_values, expected_stats):
    normalized, stats = plotting.normalize(var, params=params)

    np.testing.assert_allclose(np.asarray(normalized), expected_values)
    assert stats == expected_stats


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


class TestMapWithSliders:
    @pytest.mark.parametrize(
        ["sliders", "expected"],
        (
            pytest.param([ipywidgets.VBox()], [ipywidgets.VBox()], id="sliders"),
            pytest.param([], [], id="empty"),
        ),
    )
    def test_sliders(self, sliders, expected) -> None:
        map_ = plotting.MapWithSliders([lonboard.Map(layers=[]), *sliders])

        assert map_.sliders == expected or isinstance(map_.sliders[0], ipywidgets.VBox)

    def test_map(self):
        base_map = lonboard.Map(layers=[])
        wrapped_map = plotting.MapWithSliders([base_map, ipywidgets.HBox()])

        assert wrapped_map.map is base_map

    def test_layers(self):
        base_map = lonboard.Map(layers=[])
        wrapped_map = plotting.MapWithSliders([base_map, ipywidgets.HBox()])

        assert wrapped_map.layers == base_map.layers
