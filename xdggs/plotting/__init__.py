from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from xdggs.plotting.colorbar import Colorbar
from xdggs.plotting.colorize import (
    ColorizeParameters,
    colorize,
    extract_colors,
    normalize,
)
from xdggs.plotting.control import ControlPanel
from xdggs.plotting.dimensions import DimensionSliders
from xdggs.plotting.map import MapWithControls
from xdggs.plotting.variables import construct_variable_chooser

if TYPE_CHECKING:
    from lonboard import BaseLayer
    from lonboard.basemap import MaplibreBasemap
    from lonboard.experimental.view import BaseView


def format_labels(values):
    if values.dtype.kind in "M":
        labels = np.datetime_as_string(values, unit="s").tolist()
    else:
        labels = np.astype(values, np.dtypes.StringDType()).tolist()

    return labels


def extract_label(arr):
    units = arr.attrs.get("units")
    long_name = arr.attrs.get("long_name")
    standard_name = arr.attrs.get("standard_name")
    name = arr.name

    label = long_name or standard_name or name or "(unknown)"

    if units is not None:
        label += f" [{units}]"

    return label


def available_dims(arr_dims, all_dims):
    return {dim: dim in arr_dims for dim in all_dims}


@dataclass
class Container:
    widget: MapWithControls
    layer: BaseLayer
    obj: xr.Dataset | xr.DataArray
    colorize_params: ColorizeParameters


def on_slider_change(change, container):
    if isinstance(container.obj, xr.DataArray):
        arr = container.obj
    else:
        name = container.widget.control.variable_chooser.value
        arr = container.obj[name]

    available = change["owner"].dimension_available
    indexers = {dim: v for dim, v in change["new"].items() if available[dim]}

    if not indexers:
        # should not happen
        return

    new_slice = arr.isel(indexers)
    normalized, stats = normalize(new_slice, container.colorize_params)
    colors = colorize(normalized, container.colorize_params)

    layer = container.layer
    layer.get_fill_color = colors

    colorbar = container.widget.control.colorbar
    for name, value in stats.items():
        setattr(colorbar, name, value)


def on_variable_change(change, container):
    if isinstance(container.obj, xr.DataArray):
        # nothing to do
        return

    name = change["new"]
    arr = container.obj[name]

    sliders = container.widget.control.dimension_sliders
    sliders.dimension_available = available_dims(arr.dims, sliders.dimensions)
    indexers = {
        dim: v
        for dim, v in sliders.dimension_values.items()
        if sliders.dimension_available[dim]
    }

    new_slice = arr.isel(indexers)
    normalized, stats = normalize(new_slice, container.colorize_params)
    colors = colorize(normalized, container.colorize_params)

    layer = container.layer
    layer.get_fill_color = colors

    colorbar = container.widget.control.colorbar
    for name, value in stats.items():
        setattr(colorbar, name, value)
    colorbar.label = extract_label(arr)


def explore(
    obj: xr.Dataset | xr.DataArray,
    colorize_params: ColorizeParameters | dict[str, Any] = ColorizeParameters(),
    coords: list[str] | None = None,
    view: BaseView | None = None,
    basemap: MaplibreBasemap | None = None,
) -> MapWithControls:
    import lonboard

    map_kwargs = {}
    if view is not None:
        map_kwargs["view"] = view
    if basemap is not None:
        map_kwargs["basemap"] = basemap

    if isinstance(colorize_params, dict):
        colorize_params = ColorizeParameters.from_dict(colorize_params)

    if coords is None:
        coords = ["longitude", "latitude"]

    # guaranteed to be 1D
    cell_id_coord = obj.dggs.coord
    index = obj.dggs.index
    [cell_dim] = cell_id_coord.dims

    cell_ids = cell_id_coord.data
    grid_info = obj.dggs.grid_info

    variable_chooser = construct_variable_chooser(obj)
    if isinstance(obj, xr.Dataset) and not variable_chooser.variables:
        raise ValueError("cannot find spatial variables")

    if isinstance(obj, xr.Dataset):
        arr = obj[variable_chooser.value]
    else:
        arr = obj

    dimension_indices = {dim: 0 for dim in obj.dims if dim != cell_dim}
    initial_indexers = {d: v for d, v in dimension_indices.items() if d in arr.dims}
    initial_arr = arr.isel(initial_indexers)

    label = extract_label(arr)

    # TODO: look up single-dimensional indexes along the dimension
    dimension_coordinates = {
        dim: format_labels(obj[dim].data) for dim in obj.dims if dim in obj.coords
    }

    normalized_data, stats = normalize(initial_arr, params=colorize_params)
    colors = colorize(normalized_data, colorize_params)

    columns = {cell_id_coord.name: cell_ids}
    columns.update(
        {coord: obj.variables[coord].data for coord in coords if coord in obj.coords}
    )

    if ("longitude" in coords and "longitude" not in obj.coords) or (
        "latitude" in coords and "latitude" not in obj.coords
    ):
        lon, lat = grid_info.cell_ids2geographic(cell_ids)
        columns.update({"longitude": lon, "latitude": lat})
    columns.update({initial_arr.name or "data": initial_arr.data})

    layer = index._create_layer(cell_id_coord.name, columns, colors)

    map_ = lonboard.Map(layer, **map_kwargs)

    dimension_sliders = DimensionSliders(
        dimensions={
            dim: size - 1 for dim, size in obj.sizes.items() if dim != cell_dim
        },
        dimension_available=available_dims(arr.dims, set(obj.dims) - {cell_dim}),
        dimension_labels=dimension_coordinates,
    )
    colorbar = Colorbar(
        colors=extract_colors(colorize_params.cmap), label=label, **stats
    )
    controls = ControlPanel(
        variable_chooser=variable_chooser,
        dimension_sliders=dimension_sliders,
        colorbar=colorbar,
    )

    map_widget = MapWithControls(
        map=map_,
        control=controls,
    )

    container = Container(map_widget, layer, obj, colorize_params)

    # event handling
    dimension_sliders.observe(
        partial(on_slider_change, container=container), "dimension_values"
    )

    variable_chooser.observe(
        partial(on_variable_change, container=container), names="value"
    )

    return map_widget
