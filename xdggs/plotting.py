import numpy as np


def geoarrow2table(polygons):
    from arro3.core import Array, Schema, Table

    array = Array.from_arrow(polygons)
    field = array.field.with_name("geometry")
    schema = Schema([field])
    return Table.from_arrays([array], schema=schema)


def column_from_numpy(arr):
    from arro3.core import Array, ChunkedArray

    return ChunkedArray([Array.from_numpy(arr)])


def explore(
    arr,
    cell_dim="cells",
    cmap="viridis",
    center=None,
    alpha=None,
):
    import lonboard
    from lonboard import SolidPolygonLayer
    from lonboard.colormap import apply_continuous_cmap
    from matplotlib import colormaps
    from matplotlib.colors import CenteredNorm, Normalize

    if len(arr.dims) != 1 or cell_dim not in arr.dims:
        raise ValueError(
            f"exploration only works with a single dimension ('{cell_dim}')"
        )

    name = arr.name or "data"

    cell_ids = arr.dggs.coord.data
    grid_info = arr.dggs.grid_info

    polygons = grid_info.cell_boundaries(cell_ids, backend="geoarrow")

    var = arr.variable

    if center is None:
        vmin = var.min(skipna=True)
        vmax = var.max(skipna=True)
        normalizer = Normalize(vmin=vmin, vmax=vmax)
    else:
        halfrange = np.abs(var).max(skipna=True)
        normalizer = CenteredNorm(vcenter=center, halfrange=halfrange)

    data = var.data
    normalized_data = normalizer(data)

    colormap = colormaps[cmap]
    colors = apply_continuous_cmap(normalized_data, colormap, alpha=alpha)

    table = geoarrow2table(polygons).append_column(
        "cell_id", column_from_numpy(cell_ids)
    )
    if "latitude" in arr.coords and "longitude" in arr.coords:
        lat = arr["latitude"].data
        lon = arr["longitude"].data
        table = table.append_column("latitude", column_from_numpy(lat)).append_column(
            "longitude", column_from_numpy(lon)
        )
    table = table.append_column(name, column_from_numpy(data))

    layer = SolidPolygonLayer(table=table, filled=True, get_fill_color=colors)

    return lonboard.Map(layer)
