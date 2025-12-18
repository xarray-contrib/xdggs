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
        name: ChunkedArray([Array.from_numpy(np.ascontiguousarray(arr.data))]),
    } | {
        coord: ChunkedArray([Array.from_numpy(arr.coords[coord].data)])
        for coord in coords
        if coord in arr.coords
    }

    fields = [array.field.with_name(name) for name, array in arrow_arrays.items()]
    schema = Schema(fields)

    return Table.from_arrays(list(arrow_arrays.values()), schema=schema)
