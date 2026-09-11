import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xdggs
from xdggs.conventions.zarr import DecoderError, Zarr
from xdggs.tests import assert_indexes_equal, requires_healpix_geo_0_4_1


def translate(mapping):
    translations = {"grid_name": "name", "level": "refinement_level"}
    return {translations.get(name, name): value for name, value in mapping.items()}


WGS84 = {
    "xdggs": {
        "name": "WGS84",
        "semimajor_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    },
    "zarr": {
        "name": "WGS84",
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    },
}


@pytest.fixture
def healpix_dataset():
    data_vars = {"data": ("healpix_index", np.arange(12) % 2 == 0)}
    attrs = {
        "zarr_conventions": [Zarr.convention_metadata],
        "dggs": {
            "name": "healpix",
            "refinement_level": 0,
            "spatial_dimension": "healpix_index",
            "ellipsoid": WGS84["zarr"],
        },
    }
    return xr.Dataset(data_vars=data_vars, attrs=attrs)


@pytest.mark.parametrize(
    ["name", "dim"], [("cell_ids", "cells"), ("zone_ids", "zones")]
)
@pytest.mark.parametrize(
    ["grid_info", "metadata_object", "cell_ids"],
    (
        (
            {"grid_name": "healpix", "level": 1, "indexing_scheme": "nested"},
            {"name": "healpix", "refinement_level": 1, "indexing_scheme": "nested"},
            np.array([3, 6, 9], dtype="uint64"),
        ),
        (
            {
                "grid_name": "healpix",
                "level": 1,
                "indexing_scheme": "nested",
                "ellipsoid": WGS84["xdggs"],
            },
            {
                "name": "healpix",
                "refinement_level": 1,
                "indexing_scheme": "nested",
                "ellipsoid": WGS84["zarr"],
            },
            np.array([3, 6, 9], dtype="uint64"),
        ),
        (
            {"grid_name": "h3", "level": 4},
            {"name": "h3", "refinement_level": 4},
            np.array([0x832830FFFFFFFFF], dtype="uint64"),
        ),
    ),
)
def test_decode(grid_info, metadata_object, cell_ids, name, dim):
    convention = Zarr()

    dggs_metadata_object = metadata_object | {
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


def test_decode_no_coordinate(healpix_dataset):
    from xarray.indexes import PandasIndex

    from xdggs.healpix import HealpixIndex, HealpixInfo

    actual = Zarr().decode(healpix_dataset, grid_info=None, name=None, index_options={})

    index = HealpixIndex(
        PandasIndex(pd.RangeIndex(12, name="cell_ids"), dim="healpix_index"),
        grid_info=HealpixInfo.from_dict(
            {
                "level": 0,
                "indexing_scheme": "nested",
                "ellipsoid": WGS84["xdggs"],
            }
        ),
        name="cell_ids",
        dim="healpix_index",
    )
    expected = healpix_dataset.drop_attrs().assign_coords(
        xr.Coordinates.from_xindex(index)
    )
    xr.testing.assert_identical(actual, expected)
    assert_indexes_equal(actual["cell_ids"].xindexes, expected["cell_ids"].xindexes)


@pytest.mark.parametrize("key", ["zarr_conventions", "dggs"])
def test_raise_decode_error_missing_convention(key, healpix_dataset):
    healpix_dataset.attrs.pop(key)
    with pytest.raises(DecoderError):
        Zarr().decode(healpix_dataset, grid_info=None, name=None, index_options={})


@pytest.mark.parametrize("key", ["name", "refinement_level", "spatial_dimension"])
def test_raise_decode_error_missing_required(key, healpix_dataset):
    healpix_dataset.attrs["dggs"].pop(key)
    with pytest.raises(DecoderError, match=key):
        Zarr().decode(healpix_dataset, grid_info=None, name=None, index_options={})


def test_raise_decode_error_no_coordinate_but_default_exists(healpix_dataset):
    # the default coordinate name is "cell_ids"
    healpix_dataset["cell_ids"] = ("healpix_index", np.arange(12))
    with pytest.raises(DecoderError, match="cell_ids"):
        Zarr().decode(healpix_dataset, grid_info=None, name=None, index_options={})


def test_raise_decode_error_coordinate_not_existing(healpix_dataset):
    healpix_dataset.attrs["dggs"]["coordinate"] = "healpix_index"
    with pytest.raises(DecoderError, match="does not exist"):
        Zarr().decode(healpix_dataset, grid_info=None, name=None, index_options={})


def test_raise_decode_error_unkown_dggs(healpix_dataset):
    healpix_dataset.attrs["dggs"]["name"] = "DUMMY"
    with pytest.raises(DecoderError, match="DUMMY"):
        Zarr().decode(healpix_dataset, grid_info=None, name=None, index_options={})


@pytest.mark.parametrize(
    ["metadata_object", "grid_info", "expected_name", "variable"],
    (
        pytest.param(
            {
                "name": "healpix",
                "refinement_level": 10,
                "indexing_scheme": "nested",
                "coordinate": "cell_ranges",
                "spatial_dimension": "cells",
                "compression": "ranges",
            },
            {"grid_name": "healpix", "level": 10, "indexing_scheme": "nested"},
            "cell_ranges",
            xr.Variable(
                ("range_index", "bounds"),
                np.array(
                    [
                        [30786325577728, 35184372088832],
                        [316659348799488, 321057395310592],
                    ]
                ),
            ),
            id="ranges-10",
        ),
        pytest.param(
            {
                "name": "healpix",
                "refinement_level": 5,
                "indexing_scheme": "nested",
                "coordinate": "compacted_cell_ids",
                "spatial_dimension": "cells",
                "compression": "compacted",
            },
            {"grid_name": "healpix", "level": 5, "indexing_scheme": "nested"},
            "compacted_cell_ids",
            xr.Variable(
                ("compacted_cells"),
                np.array(
                    [216172782113783808, 792633534417207296, 1224979098644774912],
                    dtype="uint64",
                ),
            ),
            id="compacted-5",
        ),
    ),
)
def test_decode_compression(metadata_object, grid_info, expected_name, variable):
    convention = Zarr()

    name = metadata_object["coordinate"]
    ds = xr.Dataset(
        coords={name: variable},
        attrs={
            "zarr_conventions": [convention.convention_metadata],
            "dggs": metadata_object,
        },
    )

    actual = convention.decode(
        ds, grid_info=None, name="cell_ids", index_options={"index_kind": "moc"}
    )

    var = variable.copy()
    var.attrs = grid_info
    index = xdggs.index.DGGSIndex.from_variables(
        {expected_name: var},
        options={
            "index_kind": "moc",
            "compression": metadata_object["compression"],
            "dim": "cells",
        },
    )
    expected = xr.Coordinates.from_xindex(index).to_dataset()

    xr.testing.assert_equal(actual, expected)
    assert_indexes_equal(actual.xindexes, expected.xindexes)


@pytest.mark.parametrize(
    ["name", "dim"], [("cell_ids", "cells"), ("zone_ids", "zones")]
)
@pytest.mark.parametrize(
    ["grid_info", "metadata_object", "cell_ids"],
    (
        (
            {"grid_name": "healpix", "level": 1, "indexing_scheme": "nested"},
            {"name": "healpix", "refinement_level": 1, "indexing_scheme": "nested"},
            np.array([3, 6, 9], dtype="uint64"),
        ),
        (
            {
                "grid_name": "healpix",
                "level": 1,
                "indexing_scheme": "nested",
                "ellipsoid": WGS84["xdggs"],
            },
            {
                "name": "healpix",
                "refinement_level": 1,
                "indexing_scheme": "nested",
                "ellipsoid": WGS84["zarr"],
            },
            np.array([3, 6, 9], dtype="uint64"),
        ),
        (
            {"grid_name": "h3", "level": 4},
            {"name": "h3", "refinement_level": 4},
            np.array([0x832830FFFFFFFFF], dtype="uint64"),
        ),
    ),
)
def test_encode(grid_info, metadata_object, cell_ids, name, dim):
    convention = Zarr()

    index_cls = xdggs.index.GRID_REGISTRY[grid_info["grid_name"]]
    var = xr.Variable(dim, cell_ids, grid_info)
    index = index_cls.from_variables({name: var}, options={})

    obj = xr.Dataset(coords=xr.Coordinates({name: var}, indexes={name: index}))
    orig = obj.copy(deep=True)

    coord = obj.dggs.coord
    dggs_metadata_object = metadata_object | {
        "spatial_dimension": coord.dims[0],
        "coordinate": coord.name,
        "compression": "none",
    }
    metadata = {
        "zarr_conventions": [convention.convention_metadata],
        "dggs": dggs_metadata_object,
    }
    expected = xr.Dataset(coords={coord.name: (dim, cell_ids)}, attrs=metadata)

    encoded = convention.encode(obj)

    # should not modify the original dataset
    xr.testing.assert_identical(obj, orig)

    xr.testing.assert_identical(encoded, expected)
    assert_indexes_equal(encoded.xindexes, expected.xindexes)


@pytest.mark.parametrize(
    ["metadata_object", "encoding", "grid_info", "cell_ids", "encoded"],
    (
        pytest.param(
            {
                "name": "healpix",
                "refinement_level": 10,
                "indexing_scheme": "nested",
                "coordinate": "cell_ranges",
                "spatial_dimension": "cells",
                "compression": "ranges",
            },
            {"coordinate": "cell_ranges", "compression": "ranges"},
            {"grid_name": "healpix", "level": 10, "indexing_scheme": "nested"},
            xr.Variable(
                "cells",
                np.array(
                    [
                        112,
                        113,
                        114,
                        115,
                        116,
                        117,
                        118,
                        119,
                        120,
                        121,
                        122,
                        123,
                        124,
                        125,
                        126,
                        127,
                        1152,
                        1153,
                        1154,
                        1155,
                        1156,
                        1157,
                        1158,
                        1159,
                        1160,
                        1161,
                        1162,
                        1163,
                        1164,
                        1165,
                        1166,
                        1167,
                    ],
                    dtype="uint64",
                ),
            ),
            xr.Variable(
                ("range_index", "bounds"),
                np.array(
                    [
                        [30786325577728, 35184372088832],
                        [316659348799488, 321057395310592],
                    ]
                ),
            ),
            id="ranges-10",
        ),
        pytest.param(
            {
                "name": "healpix",
                "refinement_level": 2,
                "indexing_scheme": "nested",
                "coordinate": "compacted_cell_ids",
                "spatial_dimension": "cells",
                "compression": "compacted",
            },
            {
                "coordinate": "compacted_cell_ids",
                "compression": "compacted",
                "compacted_level": 1,
            },
            {"grid_name": "healpix", "level": 2, "indexing_scheme": "nested"},
            xr.Variable(
                "cells",
                np.array([4, 5, 6, 7, 20, 21, 22, 23, 32, 33, 34, 35]),
            ),
            xr.Variable(
                ("compacted_cells"),
                np.array([216172782113783808, 792633534417207296, 1224979098644774912]),
            ),
            id="compacted_flat-2",
        ),
        pytest.param(
            {
                "name": "healpix",
                "refinement_level": 5,
                "indexing_scheme": "nested",
                "coordinate": "compacted_cell_ids",
                "spatial_dimension": "cells",
                "compression": "compacted",
            },
            {"coordinate": "compacted_cell_ids", "compression": "compacted"},
            {"grid_name": "healpix", "level": 5, "indexing_scheme": "nested"},
            xr.Variable(
                "cells",
                np.array([4, 5, 6, 7, 8, 10, 12, *range(16, 32)], dtype="uint64"),
            ),
            xr.Variable(
                ("compacted_cells"),
                np.array(
                    [
                        3377699720527872,
                        4785074604081152,
                        5910974510923776,
                        7036874417766400,
                        13510798882111488,
                    ],
                    dtype="uint64",
                ),
            ),
            marks=requires_healpix_geo_0_4_1,
            id="compacted-5",
        ),
    ),
)
def test_encode_compression(metadata_object, encoding, grid_info, cell_ids, encoded):
    convention = Zarr()

    var = cell_ids.copy()
    var.attrs = grid_info
    index = xdggs.index.DGGSIndex.from_variables(
        {"cell_ids": var},
        options={
            "index_kind": "moc",
        },
    )
    ds = xr.Dataset(
        {"data": ("cells", np.arange(cell_ids.size))},
        coords=xr.Coordinates.from_xindex(index),
    )

    name = metadata_object["coordinate"]
    expected = xr.Dataset(
        {name: coord.variable for name, coord in ds.data_vars.items()},
        coords={name: encoded},
        attrs={
            "zarr_conventions": [convention.convention_metadata],
            "dggs": metadata_object,
        },
    )

    actual = convention.encode(ds, encoding=encoding)

    xr.testing.assert_equal(actual, expected)
    assert_indexes_equal(actual.xindexes, expected.xindexes)
