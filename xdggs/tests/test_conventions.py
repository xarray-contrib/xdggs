import numpy as np
import pytest
import xarray as xr

import xdggs
from xdggs import conventions
from xdggs.conventions.cf import Cf
from xdggs.conventions.errors import DecoderError
from xdggs.conventions.xdggs import Xdggs
from xdggs.conventions.zarr import Zarr
from xdggs.tests import assert_indexes_equal


class TestXdggsConvention:
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
    def test_decode(self, grid_info, cell_ids, name, dim):
        convention = Xdggs()

        var = xr.Variable(dim, cell_ids, grid_info)
        index = xdggs.index.DGGSIndex.from_variables({name: var}, options={})
        expected = xr.Coordinates.from_xindex(index).to_dataset()

        obj = xr.Dataset(coords={name: var})
        actual = convention.decode(
            obj,
            grid_info=None,
            name=name,
            index_options={},
        )
        print(obj, actual, expected)
        xr.testing.assert_identical(actual, expected)
        assert_indexes_equal(actual[name].xindexes, expected[name].xindexes)

        obj = xr.Dataset(coords={name: (dim, cell_ids)})
        actual = convention.decode(
            obj, grid_info=grid_info, name=name, index_options={}
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
    def test_encode(self, grid_info, cell_ids, name, dim):
        convention = Xdggs()

        index_cls = xdggs.index.GRID_REGISTRY[grid_info["grid_name"]]
        var = xr.Variable(dim, cell_ids, grid_info)
        index = index_cls.from_variables({name: var}, options={})

        obj = xr.Dataset(coords=xr.Coordinates({name: var}, indexes={name: index}))

        # no-op
        encoded = convention.encode(obj)

        xr.testing.assert_identical(encoded, obj)
        assert_indexes_equal(encoded.xindexes, obj.xindexes)


class TestCfConvention:
    def translate(self, mapping):
        translations = {"grid_name": "grid_mapping_name", "level": "refinement_level"}
        return {translations.get(name, name): value for name, value in mapping.items()}

    def index_metadata(self, grid_info):
        grid_name = grid_info["grid_name"]

        return {"standard_name": f"{grid_name}_index", "units": "1"}

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
    def test_decode(self, grid_info, cell_ids, name, dim):
        convention = Cf()

        var = xr.Variable(dim, cell_ids, grid_info)
        index = xdggs.index.DGGSIndex.from_variables({name: var}, options={})
        expected = xr.Coordinates.from_xindex(index).to_dataset()

        metadata = self.index_metadata(grid_info)
        translated_grid_info = self.translate(grid_info)

        cell_id_var = xr.Variable(dim, cell_ids, metadata)
        crs_var = xr.Variable((), np.array(0, dtype="int8"), translated_grid_info)

        obj = xr.Dataset(coords={name: cell_id_var, "crs": crs_var})
        actual = convention.decode(
            obj,
            grid_info=None,
            name=name,
            index_options={},
        )
        xr.testing.assert_identical(actual, expected)
        assert_indexes_equal(actual.xindexes, expected.xindexes)

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
    def test_encode(self, grid_info, cell_ids, name, dim):
        convention = Cf()

        index_cls = xdggs.index.GRID_REGISTRY[grid_info["grid_name"]]
        var = xr.Variable(dim, cell_ids, grid_info)
        index = index_cls.from_variables({name: var}, options={})

        obj = xr.Dataset(coords=xr.Coordinates({name: var}, indexes={name: index}))

        translated_grid_info = self.translate(grid_info)
        crs = xr.Variable((), np.int8(0), translated_grid_info)
        index_var = xr.Variable(dim, cell_ids, self.index_metadata(grid_info))
        expected = xr.Coordinates(
            {"crs": crs, name: index_var}, indexes={}
        ).to_dataset()

        encoded = convention.encode(obj)

        xr.testing.assert_identical(encoded, expected)
        assert_indexes_equal(encoded.xindexes, expected.xindexes)


class TestZarrConvention:
    def translate(self, mapping):
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
    def test_decode(self, grid_info, cell_ids, name, dim):
        convention = Zarr()

        dggs_metadata_object = self.translate(grid_info) | {
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
        actual = convention.decode(
            obj,
            grid_info=None,
            name=name,
            index_options={},
        )
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
    def test_encode(self, grid_info, cell_ids, name, dim):
        convention = Zarr()

        index_cls = xdggs.index.GRID_REGISTRY[grid_info["grid_name"]]
        var = xr.Variable(dim, cell_ids, grid_info)
        index = index_cls.from_variables({name: var}, options={})

        obj = xr.Dataset(coords=xr.Coordinates({name: var}, indexes={name: index}))

        coord = obj.dggs.coord
        dggs_metadata_object = self.translate(grid_info) | {
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

        xr.testing.assert_identical(encoded, expected)
        assert_indexes_equal(encoded.xindexes, expected.xindexes)


def test_decode():
    grid_info = {"grid_name": "test", "level": 2}
    ds = xr.Dataset(coords={"cell_ids": ("cells", [0, 1], grid_info)})

    # TODO: improve unknown index message
    with pytest.raises(DecoderError, match="test"):
        ds.pipe(conventions.decode)


def test_decode_indexed():
    grid_info = {"grid_name": "test", "level": 2}
    ds = xr.Dataset(coords={"cell_ids": ("cells", [0, 1], grid_info)})
    with pytest.raises(DecoderError, match="test"):
        ds.set_xindex("cell_ids").pipe(conventions.decode)
