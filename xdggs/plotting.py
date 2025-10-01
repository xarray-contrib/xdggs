from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import ipywidgets
import numpy as np
import xarray as xr
from lonboard import Map
from lonboard.models import ViewState

if TYPE_CHECKING:
    from collections.abc import Sequence

    import traitlets
    from lonboard import BaseLayer


def on_slider_change(change, container):
    owner = change["owner"]
    dim = owner.description

    indexers = {
        slider.description: slider.value
        for slider in container.dimension_sliders.children
        if slider.description != dim
    } | {dim: change["new"]}
    new_slice = container.obj.isel(indexers)

    colors = colorize(new_slice.variable, **container.colorize_kwargs)

    layer = container.map.layers[0]
    layer.get_fill_color = colors


def link_maps(
    event: traitlets.utils.bunch.Bunch,
    other_maps: Sequence[Map] = (),
) -> None:
    if isinstance(event.get("new"), ViewState):
        for lonboard_map in other_maps:
            lonboard_map.view_state = event["new"]


@dataclass
class MapContainer:
    """container for the map, any control widgets and the data object"""

    dimension_sliders: ipywidgets.VBox
    map: Map
    obj: xr.DataArray

    colorize_kwargs: dict[str, Any]

    def render(self):
        # add any additional control widgets here
        control_box = ipywidgets.HBox([self.dimension_sliders])

        return MapWithSliders(
            [self.map, control_box], layout=ipywidgets.Layout(width="100%")
        )


class MapGrid(ipywidgets.HBox):
    def __init__(self, maps: MapWithSliders | Map):
        super().__init__(maps, layout=ipywidgets.Layout(width="100%"))

    def __or__(self, other: MapGrid | MapWithSliders | Map):
        if isinstance(other, type(self)):
            other_widgets = other.children
        else:
            other_widgets = [other]
        return type(self)(self.children + other_widgets)


class MapWithSliders(ipywidgets.VBox):
    @property
    def sliders(self):
        return self.children[1]

    @property
    def map(self):
        return self.children[0]

    def __or__(self, other: MapWithSliders | Map):
        other_map = other.map if isinstance(other, MapWithSliders) else other

        self.map.observe(partial(link_maps, other_maps=[other_map]))
        other_map.observe(partial(link_maps, other_maps=[self.map]))

        layout = ipywidgets.Layout(width="50%")

        return MapGrid([self, other], layout=layout)

    def merge(self, layers, sliders):
        all_layers = list(self.map.layers) + list(layers)
        new_map = Map(all_layers)

        slider_widgets = [self.sliders]
        if sliders:
            slider_widgets.append(sliders)

        return type(self)(
            [new_map, ipywidgets.HBox(slider_widgets)], layout=self.layout
        )

    def __and__(self, other: MapWithSliders | Map | BaseLayer):
        if isinstance(other, MapWithSliders):
            layers = other.map.layers
            sliders = other.sliders
        elif isinstance(other, Map):
            layers = other.layers
            sliders = []
        else:
            layers = [other]
            sliders = []

        return self.merge(layers, sliders)


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
    import lonboard
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

    map_ = lonboard.Map(layer)

    if not initial_indexers:
        # 1D data
        return map_

    sliders = ipywidgets.VBox(
        [
            ipywidgets.IntSlider(min=0, max=arr.sizes[dim] - 1, description=dim)
            for dim in arr.dims
            if dim != cell_dim
        ]
    )

    container = MapContainer(
        sliders,
        map_,
        arr,
        colorize_kwargs={"alpha": alpha, "center": center, "colormap": colormap},
    )

    # connect slider with map
    for slider in sliders.children:
        slider.observe(partial(on_slider_change, container=container), names="value")

    return container.render()
