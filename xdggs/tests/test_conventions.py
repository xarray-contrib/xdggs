import numpy as np
import pytest
import xarray as xr

import xdggs
from xdggs.conventions.xdggs import Xdggs


class TestXdggs:
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

        obj = xr.Dataset(coords={name: (dim, cell_ids, grid_info)})
        actual = convention.decode(
            obj,
            grid_info=None,
            name=name,
            index_options={},
        )

        assert name in actual.variables and name in actual.xindexes
        xr.testing.assert_identical(actual[name], obj[name])

        obj = xr.Dataset(coords={name: (dim, cell_ids)})
        actual = convention.decode(
            obj, grid_info=grid_info, name=name, index_options={}
        )
        expected = obj.assign_coords({name: obj[name].assign_attrs(grid_info)})

        assert name in actual.variables and name in actual.xindexes
        xr.testing.assert_identical(actual[name].variable, expected[name].variable)

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
        print(obj.xindexes)
        print(encoded.xindexes)

        xr.testing.assert_identical(encoded, obj)
        assert list(encoded.xindexes) == [name]
        assert encoded.xindexes[name].equals(index)
