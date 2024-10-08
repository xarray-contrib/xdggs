import numpy as np


def center_around_prime_meridian(lon, lat):
    # three tasks:
    # - center around the prime meridian (map to a range of [-180, 180])
    # - replace the longitude of points at the poles with the median
    #   of longitude of the other vertices
    # - cells that cross the dateline should have longitudes around 180

    # center around prime meridian
    recentered = (lon + 180) % 360 - 180

    # replace lon of pole with the median of the remaining vertices
    contains_poles = np.isin(lat, np.array([-90, 90]))
    pole_cells = np.any(contains_poles, axis=-1)
    recentered[contains_poles] = np.median(
        np.reshape(
            recentered[pole_cells[:, None] & np.logical_not(contains_poles)], (-1, 3)
        ),
        axis=-1,
    )

    # keep cells that cross the dateline centered around 180
    polygons_to_fix = np.any(recentered < -100, axis=-1) & np.any(
        recentered > 100, axis=-1
    )
    result = np.where(
        polygons_to_fix[:, None] & (recentered < 0), recentered + 360, recentered
    )

    return result


def cell_ids2vertices(cell_ids, level, indexing_scheme):
    import healpy as hp

    nest = indexing_scheme == "nested"
    nside = 2**level

    boundary_vectors = hp.boundaries(nside, cell_ids, step=1, nest=nest)
    lon, lat = np.reshape(
        hp.vec2ang(np.moveaxis(boundary_vectors, 1, -1), lonlat=True), (2, -1, 4)
    )

    lon_ = center_around_prime_meridian(lon, lat)

    return np.reshape(np.stack((lon_, lat), axis=-1), (-1, 4, 2))


def cell_ids2cell_boundaries_geoarrow(cell_ids, level, indexing_scheme):
    from arro3.core import list_array

    vertices = cell_ids2vertices(cell_ids, level, indexing_scheme)
    boundaries = np.concatenate([vertices, vertices[:, :1, :]], axis=1)

    coords = np.reshape(boundaries, (-1, 2))
    coords_per_pixel = boundaries.shape[1]
    geom_offsets = np.arange(cell_ids.size + 1, dtype="int32")
    ring_offsets = geom_offsets * coords_per_pixel

    polygon_array = list_array(geom_offsets, list_array(ring_offsets, coords))
    # We need to tag the array with extension metadata (`geoarrow.polygon`) so that Lonboard knows that this is a geospatial column.
    polygon_array_with_geo_meta = polygon_array.cast(
        polygon_array.field.with_metadata(
            {
                "ARROW:extension:name": "geoarrow.polygon",
                "ARROW:extension:metadata": '{"crs": "epsg:4326"}',
            }
        )
    )
    return polygon_array_with_geo_meta


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

    polygons = cell_ids2cell_boundaries_geoarrow(
        cell_ids, level=grid_info.resolution, indexing_scheme=grid_info.indexing_scheme
    )

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
