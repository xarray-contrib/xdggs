import numpy as np
import pytest

from xdggs import h3


@pytest.mark.parametrize("resolution", [1, 10, 15])
@pytest.mark.parametrize("dim", ["cells", "zones"])
@pytest.mark.parametrize("cell_ids", (np.arange(0, 5), np.arange(5, 10)))
def test_init(cell_ids, dim, resolution):
    index = h3.H3Index(cell_ids, dim, resolution)

    assert index._resolution == resolution
    assert index._dim == dim
    assert index._pd_index.dim == dim
    assert np.all(index._pd_index.index.values == cell_ids)
