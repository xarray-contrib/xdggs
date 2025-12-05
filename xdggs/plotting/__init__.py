from typing import Any

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


def format_labels(values):
    if values.dtype.kind in "M":
        labels = np.datetime_as_string(values, unit="s").tolist()
    else:
        labels = np.astype(values, np.dtypes.StringDType()).tolist()

    return labels


def explore(
    obj,
    colorize_params: ColorizeParameters | dict[str, Any] = ColorizeParameters(),
    coords: list[str] = None,
    view=None,
    basemap=None,
):
    import lonboard
    from lonboard import SolidPolygonLayer

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

    initial_indexers = {dim: 0 for dim in obj.dims if dim != cell_dim}
    if isinstance(obj, xr.Dataset):
        initial_arr = obj[variable_chooser.value].isel(initial_indexers)
    else:
        initial_arr = obj.isel(initial_indexers)

    label = initial_arr.attrs.get("long_name") or initial_arr.name or ""

    dimension_coordinates = {
        dim: format_labels(obj[dim].data) for dim in initial_indexers
    }

    normalized_data, stats = normalize(initial_arr, params=colorize_params)
    colors = colorize(initial_arr, colorize_params)

    table = create_arrow_table(polygons, initial_arr, coords=coords)
    layer = SolidPolygonLayer(table=table, filled=True, get_fill_color=colors)

    map_ = lonboard.Map(layer, view=view, basemap=basemap)

    sliders = [
        ipywidgets.IntSlider(
            min=0,
            max=obj.sizes[dim] - 1,
            description=dim,
            disabled=dim not in initial_arr.dims,
        )
        for dim in obj.dims
        if dim != cell_dim
    ]

    colorbar = Colorbar(
        colors=extract_colors(colorize_params.cmap), label=label, **stats
    )

    map_widget = MapWithControls(
        map=map_,
        variables=variable_chooser,
        dimensions=initial_indexers,
        coordinates=dimension_coordinates,
        sliders=sliders,
        colorbar=colorbar,
    )

    # container = MapContainer(
    #     sliders,
    #     map_,
    #     obj,
    #     colorize_kwargs={"alpha": alpha, "center": center, "colormap": colormap},
    # )

    # # connect slider with map
    # for slider in sliders.children:
    #     slider.observe(partial(on_slider_change, container=container), names="value")

    # return container.render()

    return map_widget
