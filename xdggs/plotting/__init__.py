from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import ipywidgets
import numpy as np
import xarray as xr

from xdggs.plotting.arrow import create_arrow_table
from xdggs.plotting.colorbar import Colorbar
from xdggs.plotting.colorize import (
    ColorizeParameters,
    colorize,
    extract_colors,
    normalize,
)
from xdggs.plotting.map import MapWithControls
from xdggs.plotting.variables import construct_variable_chooser

if TYPE_CHECKING:
    from lonboard import BaseLayer


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


@dataclass
class Container:
    widget: MapWithControls
    layer: BaseLayer
    obj: xr.Dataset | xr.DataArray
    colorize_params: ColorizeParameters


def on_slider_change(change, changed_dim, container):
    if isinstance(container.obj, xr.DataArray):
        arr = container.obj
    else:
        name = container.widget.variables.value
        arr = container.obj[name]

    if changed_dim not in arr.dims:
        # should not happen
        return

    indexers = {
        dim: slider.value
        for dim, slider in container.widget.sliders.items()
        if dim in arr.dims and dim != changed_dim
    } | {changed_dim: change["new"]}

    for dim, slider in container.widget.sliders.items():
        slider.disabled = dim not in arr.dims

    new_slice = arr.isel(indexers)
    normalized, stats = normalize(new_slice, container.colorize_params)
    colors = colorize(normalized, container.colorize_params)

    layer = container.layer
    layer.get_fill_color = colors

    colorbar = container.widget.colorbar
    for name, value in stats.items():
        setattr(colorbar, name, value)


def on_variable_change(change, container):
    if isinstance(container.obj, xr.DataArray):
        # nothing to do
        return

    name = change["new"]
    arr = container.obj[name]

    indexers = {
        dim: slider.value
        for dim, slider in container.widget.sliders.items()
        if dim in arr.dims
    }

    new_slice = arr.isel(indexers)
    normalized, stats = normalize(new_slice, container.colorize_params)
    colors = colorize(normalized, container.colorize_params)

    for dim, slider in container.widget.sliders.items():
        slider.disabled = dim not in arr.dims

    layer = container.layer
    layer.get_fill_color = colors

    colorbar = container.widget.colorbar
    for name, value in stats.items():
        setattr(colorbar, name, value)
    colorbar.label = extract_label(arr)


def explore(
    obj,
    colorize_params: ColorizeParameters | dict[str, Any] = ColorizeParameters(),
    coords: list[str] = None,
    view=None,
    basemap=None,
):
    import lonboard
    from lonboard import SolidPolygonLayer

    map_kwargs = {}
    if view is not None:
        map_kwargs["view"] = view
    if basemap is not None:
        map_kwargs["basemap"] = basemap

    if isinstance(colorize_params, dict):
        colorize_params = ColorizeParameters.from_dict(colorize_params)

    # guaranteed to be 1D
    cell_id_coord = obj.dggs.coord
    [cell_dim] = cell_id_coord.dims

    cell_ids = cell_id_coord.data
    grid_info = obj.dggs.grid_info

    polygons = grid_info.cell_boundaries(cell_ids, backend="geoarrow")

    variable_chooser = construct_variable_chooser(obj)
    if isinstance(obj, xr.Dataset) and not variable_chooser.options:
        raise ValueError("cannot find spatial variables")

    if isinstance(obj, xr.Dataset):
        arr = obj[variable_chooser.value]
    else:
        arr = obj

    dimension_indices = {dim: 0 for dim in obj.dims if dim != cell_dim}
    initial_indexers = {d: v for d, v in dimension_indices.items() if d in arr.dims}
    initial_arr = arr.isel(initial_indexers)

    label = extract_label(arr)

    dimension_coordinates = {dim: format_labels(obj[dim].data) for dim in obj.dims}

    normalized_data, stats = normalize(initial_arr, params=colorize_params)
    colors = colorize(normalized_data, colorize_params)

    table = create_arrow_table(polygons, initial_arr, coords=coords)
    layer = SolidPolygonLayer(table=table, filled=True, get_fill_color=colors)

    map_ = lonboard.Map(layer, **map_kwargs)

    sliders = {
        dim: ipywidgets.IntSlider(
            min=0,
            max=obj.sizes[dim] - 1,
            disabled=dim not in arr.dims,
            readout=False,
        )
        for dim in obj.dims
        if dim != cell_dim
    }

    colorbar = Colorbar(
        colors=extract_colors(colorize_params.cmap), label=label, **stats
    )

    map_widget = MapWithControls(
        map=map_,
        variables=variable_chooser,
        dimensions=dimension_indices,
        coordinates=dimension_coordinates,
        sliders=sliders,
        colorbar=colorbar,
    )

    container = Container(map_widget, layer, obj, colorize_params)

    # event handling
    for dim, slider in sliders.items():
        slider.observe(
            partial(on_slider_change, changed_dim=dim, container=container),
            names="value",
        )

    variable_chooser.observe(
        partial(on_variable_change, container=container), names="value"
    )

    return map_widget
