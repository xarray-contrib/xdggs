from dataclasses import dataclass
from functools import partial
from typing import Any

import ipywidgets
import numpy as np
import xarray as xr
from lonboard import Map


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

        return ipywidgets.VBox([self.map, control_box])


def create_arrow_table(polygons, arr, coords=None):
    from arro3.core import Array, ChunkedArray, Schema, Table

    if coords is None:
        coords = ["latitude", "longitude"]

    array = Array.from_arrow(polygons)
    name = arr.name or "data"
    arrow_arrays = {
        "geometry": array,
        "cell_ids": ChunkedArray([Array.from_numpy(arr.coords["cell_ids"])]),
        name: ChunkedArray([Array.from_numpy(arr.data)]),
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
