from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import ipywidgets
import numpy as np
import xarray as xr
from lonboard import BaseLayer
from lonboard import Map as LonboardMap


@dataclass
class Container:
    obj: xr.DataArray
    colorize_kwargs: dict[str, Any]
    layer: BaseLayer
    dimension_sliders: list[ipywidgets.IntSlider]


def on_slider_change(change, container):
    owner = change["owner"]
    dim = owner.description

    indexers = {
        slider.description: slider.value
        for slider in container.dimension_sliders
        if slider.description != dim
    } | {dim: change["new"]}

    new_slice = container.obj.isel(indexers)
    colors = colorize(new_slice.variable, **container.colorize_kwargs)

    layer = container.layer
    layer.get_fill_color = colors


def render_map(
    map_: Map, dimension_sliders: list[ipywidgets.IntSlider]
) -> Map | MapWithSliders:
    if not dimension_sliders:
        return map_

    slider_box = ipywidgets.VBox(dimension_sliders)
    control_box = ipywidgets.HBox([slider_box])

    return MapWithSliders([map_, control_box], layout=ipywidgets.Layout(width="100%"))


def extract_maps(obj: MapGrid | MapWithSliders | Map | LonboardMap):
    if isinstance(obj, (Map, LonboardMap)):
        return [obj]

    return getattr(obj, "maps", (obj.map,))


class Map(LonboardMap):
    def __or__(self, other: Map | MapWithSliders):
        if isinstance(other, MapGrid):
            return NotImplemented

        return MapGrid([self, other])

    def __and__(self, other):
        if isinstance(other, (MapWithSliders, MapGrid)):
            return NotImplemented

        if isinstance(other, BaseLayer):
            other_layers = [other]
        else:
            other_layers = list(other.layers)

        layers = list(self.layers) + list(other_layers)

        return type(self)(layers)


class MapWithSliders(ipywidgets.VBox):
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

    def __or__(self, other: MapWithSliders | Map):
        [other_map] = extract_maps(other)

        return MapGrid([self, other], synchronize=True)

    def __ror__(self, other: Map):
        [other_map] = extract_maps(other)

        return MapGrid([other_map, self], synchronize=True)

    def _merge(self, layers, sliders):
        all_layers = list(self.map.layers) + list(layers)
        new_map = Map(all_layers)

        slider_widgets = []
        if self.sliders:
            slider_widgets.extend(self.sliders)
        if sliders:
            slider_widgets.extend(sliders)

        widgets = [new_map]
        if slider_widgets:
            widgets.append(ipywidgets.HBox(slider_widgets))

        return type(self)(widgets, layout=self.layout)

    def add_layer(self, layer: BaseLayer):
        self.map.add_layer(layer)

    def __and__(self, other: MapWithSliders | Map | BaseLayer):
        if isinstance(other, MapGrid):
            return NotImplemented

        if isinstance(other, BaseLayer):
            layers = [other]
            sliders = []
        else:
            layers = other.layers
            sliders = getattr(other, "sliders", [])

        return self._merge(layers, sliders)

    def __rand__(self, other: Map | BaseLayer):
        return self & other


class MapGrid(ipywidgets.GridBox):
    def __init__(
        self,
        maps: MapWithSliders | Map = None,
        n_columns: int = 2,
        synchronize: bool = False,
    ):
        self.n_columns = n_columns
        self.synchronize = synchronize

        column_width = 100 // n_columns
        layout = ipywidgets.Layout(
            width="100%", grid_template_columns=f"repeat({n_columns}, {column_width}%)"
        )

        if maps is None:
            maps = []

        super().__init__(maps, layout=layout)

        if synchronize and maps:
            self.synchronize_maps()

    def _replace_maps(self, maps):
        return type(self)(maps, n_columns=self.n_columns, synchronize=self.synchronize)

    def add_map(self, map_: MapWithSliders | Map):
        return self._replace_maps(self.maps + (map_,))

    @property
    def maps(self):
        return self.children

    def synchronize_maps(self):
        if not self.maps:
            raise ValueError("no maps to synchronize found")

        all_maps = [getattr(m, "map", m) for m in self.maps]

        first = all_maps[0]
        for second in all_maps[1:]:
            ipywidgets.jslink((first, "view_state"), (second, "view_state"))

    def __or__(self, other: MapGrid | MapWithSliders | Map):
        other_maps = extract_maps(other)

        return self._replace_maps(self.maps + other_maps)

    def __ror__(self, other: MapWithSliders | Map):
        other_maps = extract_maps(other)

        return self._replace_maps(self.maps + other_maps)


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
    } | {
        coord: ChunkedArray([Array.from_numpy(arr.coords[coord].data)])
        for coord in coords
        if coord in arr.coords
    }

    fields = [array.field.with_name(name) for name, array in arrow_arrays.items()]
    schema = Schema(fields)

    return Table.from_arrays(list(arrow_arrays.values()), schema=schema)


def normalize(var, center=None):
    from matplotlib.colors import CenteredNorm, Normalize

    if center is None:
        vmin = var.min(skipna=True)
        vmax = var.max(skipna=True)
        normalizer = Normalize(vmin=vmin, vmax=vmax)
    else:
        halfrange = np.abs(var - center).max(skipna=True)
        normalizer = CenteredNorm(vcenter=center, halfrange=halfrange)

    return normalizer(var.data)


def colorize(var, *, center, colormap, alpha):
    from lonboard.colormap import apply_continuous_cmap

    normalized_data = normalize(var, center=center)

    return apply_continuous_cmap(normalized_data, colormap, alpha=alpha)


def explore(
    arr,
    cmap="viridis",
    center=None,
    alpha=None,
    coords=None,
):
    from lonboard import SolidPolygonLayer
    from matplotlib import colormaps

    # guaranteed to be 1D
    cell_id_coord = arr.dggs.coord
    [cell_dim] = cell_id_coord.dims

    cell_ids = cell_id_coord.data
    grid_info = arr.dggs.grid_info

    polygons = grid_info.cell_boundaries(cell_ids, backend="geoarrow")

    initial_indexers = {dim: 0 for dim in arr.dims if dim != cell_dim}
    initial_arr = arr.isel(initial_indexers)

    colormap = colormaps[cmap] if isinstance(cmap, str) else cmap
    colors = colorize(initial_arr, center=center, alpha=alpha, colormap=colormap)

    table = create_arrow_table(polygons, initial_arr, coords=coords)
    layer = SolidPolygonLayer(table=table, filled=True, get_fill_color=colors)

    map_ = LonboardMap(layer)

    sliders = [
        ipywidgets.IntSlider(min=0, max=arr.sizes[dim] - 1, description=dim)
        for dim in arr.dims
        if dim != cell_dim
    ]

    map_object = render_map(map_, sliders)

    container = Container(
        arr,
        colorize_kwargs={
            "alpha": alpha,
            "center": center,
            "colormap": colormap,
        },
        layer=layer,
        dimension_sliders=sliders,
    )

    # connect slider with map
    for slider in sliders:
        slider.observe(partial(on_slider_change, container=container), names="value")

    return map_object
