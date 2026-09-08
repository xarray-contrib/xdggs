import numpy as np
import pytest
import xarray as xr

import xdggs
from xdggs.conventions import Zarr
from xdggs.tests import assert_indexes_equal


def translate(mapping):
    translations = {"grid_name": "name", "level": "refinement_level"}
    return {translations.get(name, name): value for name, value in mapping.items()}


@pytest.mark.parametrize(
    ["name", "dim"], [("cell_ids", "cells"), ("zone_ids", "zones")]
)
@pytest.mark.parametrize(
    ["grid_info", "cell_ids"],
    (
        (
            {"grid_name": "healpix", "level": 1, "indexing_scheme": "nested"},
            np.array([3, 6, 9], dtype="uint64"),
        ),
        (
            {"grid_name": "h3", "level": 4},
            np.array([0x832830FFFFFFFFF], dtype="uint64"),
        ),
    ),
)
def test_decode(grid_info, cell_ids, name, dim):
    convention = Zarr()

    dggs_metadata_object = translate(grid_info) | {
        "spatial_dimension": dim,
        "coordinate": name,
        "compression": "none",
    }
    metadata = {
        "zarr_conventions": [convention.convention_metadata],
        "dggs": dggs_metadata_object,
    }

    var = xr.Variable(dim, cell_ids)
    index = xdggs.index.DGGSIndex.from_variables(
        {name: xr.Variable(dim, cell_ids, grid_info)}, options={}
    )
    expected = xr.Coordinates.from_xindex(index).to_dataset()

    obj = xr.Dataset(coords={name: var}, attrs=metadata)
    orig = obj.copy(deep=True)
    actual = convention.decode(
        obj,
        grid_info=None,
        name=name,
        index_options={},
    )
    # should not modify the original dataset
    xr.testing.assert_identical(obj, orig)

    print(obj, actual, expected)
    xr.testing.assert_identical(actual, expected)
    assert_indexes_equal(actual[name].xindexes, expected[name].xindexes)

    obj = xr.Dataset(coords={name: (dim, cell_ids)})
    actual = convention.decode(
        obj, grid_info=dggs_metadata_object, name=name, index_options={}
    )
    xr.testing.assert_identical(actual, expected)
    assert_indexes_equal(actual[name].xindexes, expected[name].xindexes)


@pytest.mark.parametrize(
    ["name", "dim"], [("cell_ids", "cells"), ("zone_ids", "zones")]
)
@pytest.mark.parametrize(
    ["grid_info", "cell_ids"],
    (
        (
            {"grid_name": "healpix", "level": 1, "indexing_scheme": "nested"},
            np.array([3, 6, 9], dtype="uint64"),
        ),
        (
            {"grid_name": "h3", "level": 4},
            np.array([0x832830FFFFFFFFF], dtype="uint64"),
        ),
    ),
)
def test_encode(grid_info, cell_ids, name, dim):
    convention = Zarr()

    index_cls = xdggs.index.GRID_REGISTRY[grid_info["grid_name"]]
    var = xr.Variable(dim, cell_ids, grid_info)
    index = index_cls.from_variables({name: var}, options={})

    obj = xr.Dataset(coords=xr.Coordinates({name: var}, indexes={name: index}))
    orig = obj.copy(deep=True)

    coord = obj.dggs.coord
    dggs_metadata_object = translate(grid_info) | {
        "spatial_dimension": coord.dims[0],
        "coordinate": coord.name,
        "compression": "none",
    }
    metadata = {
        "zarr_conventions": [convention.convention_metadata],
        "dggs": dggs_metadata_object,
    }
    expected = obj.drop_indexes(coord.name).assign_attrs(metadata)

    encoded = convention.encode(obj)

    # should not modify the original dataset
    xr.testing.assert_identical(obj, orig)

    xr.testing.assert_identical(encoded, expected)
    assert_indexes_equal(encoded.xindexes, expected.xindexes)
