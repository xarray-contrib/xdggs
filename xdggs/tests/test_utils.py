import pytest

import xdggs.utils


@pytest.mark.parametrize("name", ["h3", "s2", "rhealpix", "healpix"])
def test_register_dggs(monkeypatch, name):
    registry = {}

    monkeypatch.setattr(xdggs.utils, "GRID_REGISTRY", registry)

    grid = object()
    registered_grid = xdggs.utils.register_dggs(name)(grid)

    assert grid is registered_grid

    assert len(registry) == 1
    assert name in registry and registry[name] is grid
