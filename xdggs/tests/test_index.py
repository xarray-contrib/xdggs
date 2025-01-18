import pytest
import xarray as xr

from xdggs import index


@pytest.fixture
def dggs_example():
    return xr.Dataset(
        coords={"cell_ids": ("cells", [0, 1], {"grid_name": "test", "level": 2})}
    )


def test_create_index(dggs_example):
    # TODO: improve unknown index message
    with pytest.raises(ValueError, match="test"):
        dggs_example.set_xindex("cell_ids", index.DGGSIndex)


def test_decode(dggs_example):
    # TODO: improve unknown index message
    with pytest.raises(ValueError, match="test"):
        dggs_example.pipe(index.decode)


def test_decode_indexed(dggs_example):
    with pytest.raises(ValueError, match="test"):
        dggs_example.set_xindex("cell_ids").pipe(index.decode)
