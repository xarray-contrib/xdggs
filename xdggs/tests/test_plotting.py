import ipywidgets
import lonboard
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
    actual = plotting.create_arrow_table(polygons, arr, coords=coords)

    assert actual == expected


# Tests for normalize and colorize functions removed - they are now part of the Colorizer class


class TestColorizer:
    def test_for_dataarray_basic(self):
        """Test basic colorizer creation from DataArray."""
        data = xr.DataArray([0, 1, 2, 3], dims="cells")
        colorizer = plotting.Colorizer.for_dataarray(data, cmap="viridis")

        assert colorizer.colormap.name == "viridis"
        assert colorizer.normalizer.vmin == 0
        assert colorizer.normalizer.vmax == 3

    def test_for_dataarray_with_center(self):
        """Test colorizer with centered normalization."""
        data = xr.DataArray([-5, -2, 0, 2, 5], dims="cells")
        colorizer = plotting.Colorizer.for_dataarray(data, cmap="coolwarm", center=0)

        assert colorizer.colormap.name == "coolwarm"
        assert hasattr(colorizer.normalizer, "vcenter")
        assert colorizer.normalizer.vcenter == 0

    def test_for_dataarray_with_vmin_vmax(self):
        """Test colorizer with explicit vmin/vmax."""
        data = xr.DataArray([0, 1, 2, 3], dims="cells")
        colorizer = plotting.Colorizer.for_dataarray(
            data, cmap="plasma", vmin=-10, vmax=10
        )

        assert colorizer.normalizer.vmin == -10
        assert colorizer.normalizer.vmax == 10

    def test_for_dataarray_with_alpha(self):
        """Test colorizer with alpha transparency."""
        data = xr.DataArray([0, 1, 2, 3], dims="cells")
        colorizer = plotting.Colorizer.for_dataarray(data, cmap="viridis", alpha=0.5)

        assert colorizer.alpha == 0.5

    def test_for_dataarray_robust(self):
        """Test robust normalization using percentiles."""
        data = xr.DataArray([0, 1, 2, 3, 100], dims="cells")  # outlier at 100
        colorizer = plotting.Colorizer.for_dataarray(data, cmap="viridis", robust=True)

        # With robust=True, should use 2nd and 98th percentiles
        assert colorizer.normalizer.vmin < 1
        assert colorizer.normalizer.vmax < 100

    def test_for_dataset_basic(self):
        """Test colorizer creation from Dataset."""
        ds = xr.Dataset({"temperature": xr.DataArray([10, 20, 30], dims="cells")})
        colorizer = plotting.Colorizer.for_dataset(
            "temperature", ds["temperature"], cmap="viridis"
        )

        assert colorizer.colormap.name == "viridis"
        assert colorizer.normalizer.vmin == 10
        assert colorizer.normalizer.vmax == 30

    def test_for_dataset_with_dict_cmap(self):
        """Test dataset colorizer with dictionary of colormaps."""
        data = xr.DataArray([10, 20, 30], dims="cells")
        colorizer = plotting.Colorizer.for_dataset(
            "temperature", data, cmap={"temperature": "coolwarm", "pressure": "viridis"}
        )

        assert colorizer.colormap.name == "coolwarm"

    def test_for_dataset_with_dict_center(self):
        """Test dataset colorizer with dictionary of center values."""
        data = xr.DataArray([-5, 0, 5], dims="cells")
        colorizer = plotting.Colorizer.for_dataset(
            "temperature",
            data,
            cmap="coolwarm",
            center={"temperature": 0, "pressure": 1000},
        )

        assert colorizer.normalizer.vcenter == 0

    def test_colorize_returns_array(self):
        """Test that colorize returns a numpy array."""
        data = xr.DataArray([0, 1, 2, 3], dims="cells")
        colorizer = plotting.Colorizer.for_dataarray(data, cmap="viridis")

        colors = colorizer.colorize(data.variable)

        assert isinstance(colors, np.ndarray)
        assert colors.shape[0] == 4  # 4 cells
        assert colors.shape[1] in [3, 4]  # RGB or RGBA

    def test_colorize_with_alpha(self):
        """Test that colorize with alpha returns RGBA."""
        data = xr.DataArray([0, 1, 2, 3], dims="cells")
        colorizer = plotting.Colorizer.for_dataarray(data, cmap="viridis", alpha=0.8)

        colors = colorizer.colorize(data.variable)

        assert colors.shape[1] == 4  # RGBA
        assert colors.dtype == np.uint8

    def test_get_cmap_preview(self):
        """Test colormap preview generation."""
        data = xr.DataArray([0, 1, 2, 3], dims="cells")
        colorizer = plotting.Colorizer.for_dataarray(data, cmap="viridis")

        fig, ax = colorizer.get_cmap_preview("Test Label")

        assert fig is not None
        assert ax is not None

    @pytest.mark.parametrize(
        ["data", "kwargs", "expected"],
        (
            pytest.param(
                xr.DataArray([0, 3], dims="cells"),
                {"cmap": "viridis", "center": 2, "alpha": 1},
                np.array([[68, 1, 84], [94, 201, 97]], dtype="uint8"),
                id="centered-rgb",
            ),
            pytest.param(
                xr.DataArray([-1, 1], dims="cells"),
                {"cmap": "viridis", "center": None, "alpha": 0.8},
                np.array([[68, 1, 84, 204], [253, 231, 36, 204]], dtype="uint8"),
                id="linear-rgba",
            ),
        ),
    )
    def test_colorize_expected_values(self, data, kwargs, expected):
        """Test colorize produces expected color arrays for specific inputs."""
        colorizer = plotting.Colorizer.for_dataarray(data, **kwargs)
        actual = colorizer.colorize(data.variable)

        np.testing.assert_equal(actual, expected)


class TestMapContainer:
    def test_init(self):
        """Test MapContainer initialization with proper map and data."""
        from arro3.core import Table
        from lonboard import SolidPolygonLayer

        # Create a valid layer with geometry
        obj = xr.DataArray(
            [[0, 1], [2, 3]],
            coords={"cell_ids": ("cells", [10, 26])},
            dims=["time", "cells"],
        ).dggs.decode({"grid_name": "healpix", "level": 1, "indexing_scheme": "nested"})

        # Get polygons from grid
        cell_ids = obj.dggs.coord.data
        grid_info = obj.dggs.grid_info
        polygons = grid_info.cell_boundaries(cell_ids, backend="geoarrow")
        table = plotting.create_arrow_table(polygons, obj.isel(time=0))
        layer = SolidPolygonLayer(table=table)

        map_ = lonboard.Map(layers=[layer])
        colorizer_kwargs = {"cmap": "viridis", "alpha": 0.8}

        container = plotting.MapContainer(
            map_=map_,
            obj=obj,
            colorizer_kwargs=colorizer_kwargs,
        )

        assert container.map == map_
        xr.testing.assert_equal(container.obj, obj)
        assert container.colorizer_kwargs == colorizer_kwargs

    def test_render(self):
        """Test MapContainer render method."""
        from lonboard import SolidPolygonLayer

        obj = xr.DataArray(
            [[0, 1], [2, 3]],
            coords={"cell_ids": ("cells", [10, 26])},
            dims=["time", "cells"],
        ).dggs.decode({"grid_name": "healpix", "level": 1, "indexing_scheme": "nested"})

        # Get polygons from grid
        cell_ids = obj.dggs.coord.data
        grid_info = obj.dggs.grid_info
        polygons = grid_info.cell_boundaries(cell_ids, backend="geoarrow")
        table = plotting.create_arrow_table(polygons, obj.isel(time=0))
        layer = SolidPolygonLayer(table=table)

        map_ = lonboard.Map(layers=[layer])
        colorizer_kwargs = {"cmap": "viridis"}

        container = plotting.MapContainer(
            map_=map_,
            obj=obj,
            colorizer_kwargs=colorizer_kwargs,
        )
        rendered = container.render()

        assert isinstance(rendered, plotting.MapWithControls)


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


class TestMapWithControls:
    @pytest.mark.parametrize(
        ["sliders", "expected"],
        (
            pytest.param([ipywidgets.VBox()], [ipywidgets.VBox()], id="sliders"),
            pytest.param([], [], id="empty"),
        ),
    )
    def test_sliders(self, sliders, expected) -> None:
        map_ = plotting.MapWithControls([lonboard.Map(layers=[]), *sliders])

        assert map_.sliders == expected or isinstance(map_.sliders[0], ipywidgets.VBox)

    def test_map(self):
        base_map = lonboard.Map(layers=[])
        wrapped_map = plotting.MapWithControls([base_map, ipywidgets.HBox()])

        assert wrapped_map.map is base_map

    def test_layers(self):
        base_map = lonboard.Map(layers=[])
        wrapped_map = plotting.MapWithControls([base_map, ipywidgets.HBox()])

        assert wrapped_map.layers == base_map.layers
