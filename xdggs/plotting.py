import numpy as np


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
    cell_dim="cells",
    cmap="viridis",
    center=None,
    alpha=None,
):
    import lonboard
    from lonboard import SolidPolygonLayer
    from matplotlib import colormaps

    if cell_dim not in arr.dims:
        raise ValueError(
            f"exploration plotting only works with a spatial dimension ('{cell_dim}')"
        )

    cell_ids = arr.dggs.coord.data
    grid_info = arr.dggs.grid_info

    polygons = grid_info.cell_boundaries(cell_ids, backend="geoarrow")

    colormap = colormaps[cmap] if isinstance(cmap, str) else cmap
    colors = colorize(arr.variable, center=center, alpha=alpha, colormap=colormap)

    table = create_arrow_table(polygons, arr)
    layer = SolidPolygonLayer(table=table, filled=True, get_fill_color=colors)

    return lonboard.Map(layer)
