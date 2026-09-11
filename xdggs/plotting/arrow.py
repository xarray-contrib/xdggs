import numpy as np
import numpy.typing as npt
from arro3.core import Array, ChunkedArray, Schema, Table


def create_arrow_table(columns: dict[str, npt.NDArray]) -> Table:
    def _arrow_import(arr):
        if isinstance(arr, Array):
            return arr
        elif isinstance(arr, np.ndarray):
            return ChunkedArray([Array.from_numpy(np.ascontiguousarray(arr))])
        else:
            raise NotImplementedError(
                f"unknown array type: {type(arr)}, don't know how to convert that to arrow."
            )

    arrow_arrays = {name: _arrow_import(data) for name, data in columns.items()}

    fields = [array.field.with_name(name) for name, array in arrow_arrays.items()]
    schema = Schema(fields)

    return Table.from_arrays(list(arrow_arrays.values()), schema=schema)
