import numpy as np


def explore(
    arr,
    cell_boundaries="cell_boundaries",
    cell_dim="cells",
    cmap="viridis",
    center=None,
):
    import geopandas as gpd
    import lonboard
    from lonboard import SolidPolygonLayer
    from lonboard.colormap import apply_continuous_cmap
    from matplotlib import colormaps
    from matplotlib.colors import CenteredNorm, Normalize

    if len(arr.dims) != 1 or cell_dim not in arr.dims:
        raise ValueError(
            f"exploration only works with a single dimension ('{cell_dim}')"
        )

    if cell_boundaries not in arr.coords:
        raise ValueError(
            f"cannot find the cell boundaries coordinate: '{cell_boundaries}'"
        )

    name = arr.name or "__temporary_name__"

    gdf = (
        arr.to_dataset(name=name)
        .to_pandas()
        .pipe(gpd.GeoDataFrame, geometry=cell_boundaries, crs=4326)
    )

    data = gdf[name]

    if center is None:
        normalizer = Normalize()
    else:
        halfrange = np.abs(data).max()
        normalizer = CenteredNorm(vcenter=center, halfrange=halfrange)

    normalized_data = normalizer(data)

    colormap = colormaps[cmap]
    colors = apply_continuous_cmap(normalized_data, colormap)

    layer = SolidPolygonLayer.from_geopandas(gdf, filled=True, get_fill_color=colors)

    return lonboard.Map(layer)
