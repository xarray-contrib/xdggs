from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING

import ipywidgets
import numpy as np
import xarray as xr
from lonboard import BaseLayer, Map

from xdggs.h3 import H3Info

if TYPE_CHECKING:
    from lonboard import Map as LonboardMap
    from matplotlib.colors import CenteredNorm, Colormap, Normalize


@dataclass
class Colorizer:
    colormap: Colormap
    normalizer: CenteredNorm | Normalize
    alpha: float | None = None

    @staticmethod
    def _get_normalizer(
        data,
        center: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = False,
    ) -> CenteredNorm | Normalize:
        from matplotlib.colors import CenteredNorm, Normalize

        # Logic: If one or both of vmin and vmax are set, use them.
        # If one is not set, compute it from the data depending on robust flag.
        # If neither is set, try to use center if provided.
        # If center is not provided, use min and max of data, depending on robust flag.
        # Robust flag means using the 2nd and 98th percentiles instead of min and max.
        if vmin is not None or vmax is not None:
            if vmin is None:
                if robust:
                    vmin = np.nanpercentile(data, 2)
                else:
                    vmin = np.nanmin(data)
            if vmax is None:
                if robust:
                    vmax = np.nanpercentile(data, 98)
                else:
                    vmax = np.nanmax(data)
            normalizer = Normalize(vmin=vmin, vmax=vmax)
        elif center is not None:
            if robust:
                halfrange = np.abs(data - center).quantile(0.98)
            else:
                halfrange = np.abs(data - center).max(skipna=True)
            normalizer = CenteredNorm(vcenter=center, halfrange=halfrange)
        else:
            if robust:
                vmin = np.nanpercentile(data, 2)
                vmax = np.nanpercentile(data, 98)
            else:
                vmin = np.nanmin(data)
                vmax = np.nanmax(data)
            normalizer = Normalize(vmin=vmin, vmax=vmax)

        return normalizer

    @classmethod
    def for_dataset(
        cls,
        var_name: str,
        data: xr.DataArray,
        cmap: str | Colormap | dict[str, str | Colormap] = "viridis",
        alpha: float | None = None,
        center: float | dict[str, float] | None = None,
        vmin: float | dict[str, float] | None = None,
        vmax: float | dict[str, float] | None = None,
        robust: bool = False,
    ):
        from matplotlib import colormaps

        if isinstance(cmap, dict):
            current_cmap = cmap.get(var_name, "viridis")
        else:
            current_cmap = cmap
        if isinstance(center, dict):
            current_center = center.get(var_name, None)
        else:
            current_center = center
        if isinstance(vmin, dict):
            current_vmin = vmin.get(var_name, None)
        else:
            current_vmin = vmin
        if isinstance(vmax, dict):
            current_vmax = vmax.get(var_name, None)
        else:
            current_vmax = vmax

        colormap = colormaps[current_cmap] if isinstance(current_cmap, str) else current_cmap

        normalizer = cls._get_normalizer(
            data,
            center=current_center,
            vmin=current_vmin,
            vmax=current_vmax,
            robust=robust,
        )

        return cls(
            colormap=colormap,
            normalizer=normalizer,
            alpha=alpha,
        )

    @classmethod
    def for_dataarray(
        cls,
        data: xr.DataArray,
        cmap: str | Colormap = "viridis",
        alpha: float | None = None,
        center: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = False,
    ):
        from matplotlib import colormaps

        colormap = colormaps[cmap] if isinstance(cmap, str) else cmap

        normalizer = cls._get_normalizer(
            data,
            center=center,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
        )

        return cls(
            colormap=colormap,
            normalizer=normalizer,
            alpha=alpha,
        )

    def get_cmap_preview(self, label: str):
        import matplotlib.pyplot as plt

        sm = plt.cm.ScalarMappable(cmap=self.colormap, norm=self.normalizer)
        fig, ax = plt.subplots(figsize=(9, 0.25))
        fig.colorbar(sm, cax=ax, orientation="horizontal", label=label)
        return fig, ax

    def colorize(self, data):
        from lonboard.colormap import apply_continuous_cmap

        normalized_data = self.normalizer(data)

        return apply_continuous_cmap(
            normalized_data,
            self.colormap,
            alpha=self.alpha,
        )


def create_slider_widget(arr, dim):
    # If the dimension has coordinates, use them as labels
    # Otherwise, use integer indices
    style = {"description_width": "0px"}
    layout = ipywidgets.Layout(min_width="300px")

    if dim in arr.coords:
        # Use a Float Slider for numeric coordinates
        # Use a Select Slider for non-numeric coordinates, e.g. time or strings
        coord_values = arr.coords[dim].data
        if np.issubdtype(coord_values.dtype, np.number):
            slider = ipywidgets.FloatSlider(
                min=float(coord_values.min()),
                max=float(coord_values.max()),
                step=float(np.diff(np.unique(coord_values)).min()),
                description=dim,
                continuous_update=False,
                style=style,
                layout=layout,
            )
        else:
            # TODO: Use format for datetime display?
            slider = ipywidgets.SelectionSlider(
                options=list(coord_values),
                description=dim,
                continuous_update=False,
                style=style,
                layout=layout,
            )

    else:
        slider = ipywidgets.IntSlider(
            min=0,
            max=arr.sizes[dim] - 1,
            description=dim,
            continuous_update=False,
            style=style,
            layout=layout,
        )

    return slider


class SliderPlayer:
    """Manages play/pause functionality for a single slider."""

    def __init__(self, slider: ipywidgets.Widget, dim: str, interval: float = 0.5):
        """
        Initialize a slider player.

        Parameters
        ----------
        slider : ipywidgets.Widget
            The slider widget to control (IntSlider, FloatSlider, or SelectionSlider)
        dim : str
            The dimension name
        interval : float
            Time in seconds between steps when playing
        """
        self.slider = slider
        self.dim = dim
        self.interval = interval
        self.is_playing = False
        self._thread = None
        self._stop_event = threading.Event()

        # Create play/pause button
        self.play_button = ipywidgets.Button(
            description="▶",
            layout=ipywidgets.Layout(width="40px"),
            tooltip="Play/Pause",
        )
        self.play_button.on_click(self._toggle_play)

    def _toggle_play(self, button):
        """Toggle between play and pause states."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        """Start playing through slider values."""
        if self.is_playing:
            return

        self.is_playing = True
        self.play_button.description = "⏸"
        self.play_button.tooltip = "Pause"
        self._stop_event.clear()

        # Start the animation thread
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def pause(self):
        """Pause the animation."""
        if not self.is_playing:
            return

        self.is_playing = False
        self.play_button.description = "▶"
        self.play_button.tooltip = "Play"
        self._stop_event.set()

        # Wait for thread to finish
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _animate(self):
        """Animation loop that runs in a separate thread."""
        while not self._stop_event.is_set():
            # Get current value and determine next value
            if isinstance(self.slider, ipywidgets.IntSlider):
                current = self.slider.value
                if current >= self.slider.max:
                    self.slider.value = self.slider.min
                else:
                    self.slider.value = current + self.slider.step
            elif isinstance(self.slider, ipywidgets.FloatSlider):
                current = self.slider.value
                if current >= self.slider.max:
                    self.slider.value = self.slider.min
                else:
                    self.slider.value = min(current + self.slider.step, self.slider.max)
            elif isinstance(self.slider, ipywidgets.SelectionSlider):
                current_index = self.slider.index
                if current_index >= len(self.slider.options) - 1:
                    self.slider.index = 0
                else:
                    self.slider.index = current_index + 1

            # Wait for the specified interval
            self._stop_event.wait(self.interval)

    def widget(self):
        """Return a widget with the slider and play button."""
        return ipywidgets.VBox(
            [
                ipywidgets.Label(value=self.dim, style={"font_weight": "bold"}),
                ipywidgets.HBox(
                    [self.play_button, self.slider], layout=ipywidgets.Layout(align_items="center", gap="5px")
                ),
            ]
        )


def create_slider_with_player(arr, dim, interval: float = 0.5):
    """
    Create a slider widget with play/pause controls.

    Parameters
    ----------
    arr : xr.DataArray
        The data array containing the dimension
    dim : str
        The dimension name
    interval : float
        Time in seconds between steps when playing

    Returns
    -------
    SliderPlayer
        A slider player instance with play/pause controls
    """
    slider = create_slider_widget(arr, dim)
    return SliderPlayer(slider, dim=dim, interval=interval)


class MapContainer:
    """Container for the map, any control widgets and the data object."""

    def __init__(
        self,
        map_: LonboardMap,
        obj: xr.DataArray | xr.Dataset,
        colorizer_kwargs: dict,
        play_interval: float = 0.5,
    ):
        self.map = map_
        self.obj = obj
        self.colorizer_kwargs = colorizer_kwargs
        self.play_interval = play_interval

        cell_id_coord = self.obj.dggs.coord
        [cell_dim] = cell_id_coord.dims
        self.cell_dim = cell_dim

        self.dvar_selector = None
        if isinstance(obj, xr.Dataset):
            self.dvar_selector = ipywidgets.Dropdown(
                options=list(obj.data_vars),
                description="Variable",
                continuous_update=False,
            )
            self.dvar_selector.observe(self.create_sliders, names="value")

        # This creates self.colorizer, self.dimension_sliders, self.dimension_indexers, self.dimension_selectors
        self.create_sliders(None)
        # Quick check so that future changes to the code will fail if these attributes are missing
        assert hasattr(self, "data_label")
        assert hasattr(self, "colorizer")
        assert hasattr(self, "dimension_sliders")
        assert hasattr(self, "slider_players")
        assert hasattr(self, "dimension_indexers")
        assert hasattr(self, "dimension_selectors")
        assert hasattr(self, "control_box")

    def _get_colorizer(self, data: xr.DataArray):
        if isinstance(self.obj, xr.Dataset):
            assert self.dvar_selector is not None
            selected_var = self.dvar_selector.value
            colorizer = Colorizer.for_dataset(selected_var, data, **self.colorizer_kwargs)
        else:
            colorizer = Colorizer.for_dataarray(data, **self.colorizer_kwargs)
        return colorizer

    def _get_arr(self):
        if isinstance(self.obj, xr.Dataset):
            assert self.dvar_selector is not None
            selected_var = self.dvar_selector.value
            arr = self.obj[selected_var]
        else:
            arr = self.obj
        return arr

    def create_sliders(self, change):
        # Pause any existing players before recreating sliders
        if hasattr(self, "slider_players"):
            for player in self.slider_players.values():
                player.pause()

        arr = self._get_arr()

        # Update the label information
        if "long_name" in arr.attrs:
            self.data_label = arr.attrs["long_name"]
        else:
            self.data_label = arr.name or "data"
        if "units" in arr.attrs:
            self.data_label += f" ({arr.attrs['units']})"

        # Update the colorizer
        self.colorizer = self._get_colorizer(arr)

        # Update sliders based on the new variable's dimensions
        # Create slider players for dimensions with more than one value
        self.slider_players = {
            dim: create_slider_with_player(arr, dim, interval=self.play_interval)
            for dim in arr.dims
            if dim != self.cell_dim and arr.sizes[dim] > 1
        }

        # Store reference to the actual sliders for easier access
        self.dimension_sliders = {dim: player.slider for dim, player in self.slider_players.items()}

        # Reset indexers and selectors
        self.dimension_indexers = {
            dim: 0 for dim, slider in self.dimension_sliders.items() if isinstance(slider, ipywidgets.IntSlider)
        }
        self.dimension_selectors = {
            dim: slider.value
            for dim, slider in self.dimension_sliders.items()
            if not isinstance(slider, ipywidgets.IntSlider)
        }

        # Reconnect slider change events
        for slider in self.dimension_sliders.values():
            slider.observe(partial(self.recolorize), names="value")

        self.recolorize(arr=arr)
        self.create_control_box()

    def recolorize(self, change=None, arr=None):
        if arr is None:
            arr = self._get_arr()

        if change is not None:
            dim = change["owner"].description
            if dim in self.dimension_indexers:
                self.dimension_indexers[dim] = change["new"]
            else:
                self.dimension_selectors[dim] = change["new"]
        if not self.dimension_indexers and not self.dimension_selectors:
            # No indexing needed
            new_slice = arr
        else:
            new_slice = arr.isel(self.dimension_indexers).sel(self.dimension_selectors)
        colors = self.colorizer.colorize(new_slice.variable)
        layer = self.map.layers[0]
        layer.get_fill_color = colors

    def create_control_box(self):
        import matplotlib.pyplot as plt

        control_widgets = []
        if self.dvar_selector is not None:
            control_widgets.append(self.dvar_selector)
        if len(self.slider_players):
            # Create widgets with play buttons for each slider
            slider_widgets = [player.widget() for player in self.slider_players.values()]
            control_widgets.append(ipywidgets.VBox(slider_widgets, layout={"padding": "0 10px", "margin": "0 10px"}))

        fig, _ax = self.colorizer.get_cmap_preview(self.data_label)
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        colorbar_widget = ipywidgets.Image(value=buf.read(), format="png")
        buf.close()
        plt.close(fig)

        # Create layout: controls on left, colorbar on right (wraps to new row if needed)
        controls_box = ipywidgets.HBox(
            control_widgets,
            layout=ipywidgets.Layout(flex="0 1 auto", min_width="fit-content", align_items="flex-start"),
        )
        colorbar_box = ipywidgets.Box(
            [colorbar_widget],
            layout=ipywidgets.Layout(
                flex="0 0 auto",
                align_items="center",
                max_width="500px",
                overflow="visible",
            ),
        )

        box_children = [controls_box, colorbar_box]

        if not hasattr(self, "control_box"):
            # First time creation
            self.control_box = ipywidgets.HBox(
                box_children,
                layout=ipywidgets.Layout(
                    width="100%",
                    height="auto",
                    align_items="flex-start",
                    padding="5px 0px 0px 0px",
                    flex_flow="row wrap",
                    justify_content="space-between",
                    overflow="visible",
                ),
            )
        else:
            # Empty the existing box and refill
            self.control_box.children = box_children

    def stop_all_players(self):
        """Stop all slider players."""
        if hasattr(self, "slider_players"):
            for player in self.slider_players.values():
                player.pause()

    def render(self):
        return MapWithControls(
            [self.map, self.control_box],
            layout=ipywidgets.Layout(width="100%", overflow="hidden"),
        )


def extract_maps(obj: MapGrid | MapWithControls | Map):
    if isinstance(obj, Map):
        return (obj,)

    return getattr(obj, "maps", (obj.map,))


class MapGrid(ipywidgets.GridBox):
    def __init__(
        self,
        maps: MapWithControls | Map = None,
        n_columns: int = 2,
        synchronize: bool = False,
    ):
        self.n_columns = n_columns
        self.synchronize = synchronize

        column_width = 100 // n_columns
        layout = ipywidgets.Layout(width="100%", grid_template_columns=f"repeat({n_columns}, {column_width}%)")

        if maps is None:
            maps = []

        if synchronize and maps:
            all_maps = [getattr(m, "map", m) for m in maps]

            first = all_maps[0]
            for second in all_maps[1:]:
                ipywidgets.jslink((first, "view_state"), (second, "view_state"))

        super().__init__(maps, layout=layout)

    def _replace_maps(self, maps):
        return type(self)(maps, n_columns=self.n_columns, synchronize=self.synchronize)

    def add_map(self, map_: MapWithControls | Map):
        return self._replace_maps(self.maps + (map_,))

    @property
    def maps(self):
        return self.children

    def __or__(self, other: MapGrid | MapWithControls | Map):
        other_maps = extract_maps(other)

        return self._replace_maps(self.maps + other_maps)

    def __ror__(self, other: MapWithControls | Map):
        other_maps = extract_maps(other)

        return self._replace_maps(self.maps + other_maps)


class MapWithControls(ipywidgets.VBox):
    def change_layout(self, layout):
        return type(self)(self.children, layout=layout)

    @property
    def sliders(self) -> list:
        return list(self.children[1:]) if len(self.children) > 1 else []

    @property
    def map(self) -> Map:
        return self.children[0]

    @property
    def layers(self) -> list[BaseLayer]:
        return self.map.layers

    def __or__(self, other: MapWithControls | Map):
        # [other_map] = extract_maps(other)

        return MapGrid([self, other], synchronize=True)

    def _merge(self, layers, sliders):
        all_layers = list(self.map.layers) + list(layers)
        new_map = Map(all_layers)

        slider_widgets = []
        if self.sliders:
            slider_widgets.extend(self.sliders)
        if sliders:
            slider_widgets.extend(sliders)

        widget_list = [new_map]
        if slider_widgets:
            widget_list.append(ipywidgets.HBox(slider_widgets))

        return type(self)(widget_list, layout=self.layout)

    def add_layer(self, layer: BaseLayer):
        self.map.add_layer(layer)

    def __and__(self, other: MapWithControls | Map | BaseLayer):
        if isinstance(other, BaseLayer):
            layers = [other]
            sliders = []
        else:
            layers = other.layers
            sliders = getattr(other, "sliders", [])

        return self._merge(layers, sliders)


def create_arrow_table(polygons, arr, coords=None):
    from arro3.core import Array, ChunkedArray, Schema, Table

    if coords is None:
        coords = ["latitude", "longitude"]

    array = Array.from_arrow(polygons)
    name = arr.name or "data"
    arrow_arrays = {
        "geometry": array,
        "cell_ids": ChunkedArray([Array.from_numpy(arr.coords["cell_ids"])]),
        name: ChunkedArray([Array.from_numpy(np.ascontiguousarray(arr.data))]),
    } | {coord: ChunkedArray([Array.from_numpy(arr.coords[coord].data)]) for coord in coords if coord in arr.coords}

    fields = [array.field.with_name(name) for name, array in arrow_arrays.items()]
    schema = Schema(fields)

    return Table.from_arrays(list(arrow_arrays.values()), schema=schema)


def explore(
    obj: xr.DataArray | xr.Dataset,
    coords: float | None = None,
    cmap: str | Colormap | dict[str, str | Colormap] = "viridis",
    alpha: float | None = None,
    center: float | dict[str, float] | None = None,
    vmin: float | dict[str, float] | None = None,
    vmax: float | dict[str, float] | None = None,
    robust: bool = False,
    play_interval: float = 0.5,
    **map_kwargs,
):
    import lonboard
    from lonboard import H3HexagonLayer, SolidPolygonLayer

    # guaranteed to be 1D
    cell_id_coord = obj.dggs.coord
    [cell_dim] = cell_id_coord.dims

    cell_ids = cell_id_coord.data
    grid_info = obj.dggs.grid_info

    polygons = grid_info.cell_boundaries(cell_ids, backend="geoarrow")

    if isinstance(obj, xr.Dataset):
        # pick first data variable
        first_var = next(iter(obj.data_vars))
        arr = obj[first_var]
        colorizer = Colorizer.for_dataset(
            var_name=first_var,
            data=arr,
            cmap=cmap,
            alpha=alpha,
            center=center,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
        )
    else:
        assert not isinstance(cmap, dict), "cmap cannot be a dict when obj is a DataArray"
        assert not isinstance(center, dict), "center cannot be a dict when obj is a DataArray"
        assert not isinstance(vmin, dict), "vmin cannot be a dict when obj is a DataArray"
        assert not isinstance(vmax, dict), "vmax cannot be a dict when obj is a DataArray"
        arr = obj
        colorizer = Colorizer.for_dataarray(
            data=arr,
            cmap=cmap,
            alpha=alpha,
            center=center,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
        )

    initial_indexers = {dim: 0 for dim in arr.dims if dim != cell_dim}
    initial_arr = arr.isel(initial_indexers)

    fill_colors = colorizer.colorize(initial_arr.variable)
    table = create_arrow_table(polygons, initial_arr, coords=coords)

    # Use the H3 Layer for H3 grid
    if isinstance(grid_info, H3Info):
        layer = H3HexagonLayer(
            table=table,
            get_hexagon=table["cell_ids"],
            filled=True,
            get_fill_color=fill_colors,
        )
    else:
        layer = SolidPolygonLayer(table=table, filled=True, get_fill_color=fill_colors)

    map_ = lonboard.Map(layer, **map_kwargs)

    if not initial_indexers and (isinstance(arr, xr.DataArray) or len(arr.data_vars) == 1):
        # 1D data, special case, no sliders / selectors - no interactivity needed
        # This also results in a missing colorbar, since only the raw map is returned
        return map_

    container = MapContainer(
        map_,
        obj,
        {
            "cmap": cmap,
            "alpha": alpha,
            "center": center,
            "vmin": vmin,
            "vmax": vmax,
            "robust": robust,
        },
        play_interval=play_interval,
    )

    return container.render()
