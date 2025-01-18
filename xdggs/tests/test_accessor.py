import pytest
import xarray as xr

import xdggs


@pytest.mark.parametrize(
    ["obj", "grid_info", "name"],
    (
        pytest.param(
            xr.Dataset(
                coords={
                    "cell_ids": (
                        "cells",
                        [1],
                        {
                            "grid_name": "healpix",
                            "level": 1,
                            "indexing_scheme": "ring",
                        },
                    )
                }
            ),
            None,
            None,
            id="dataset-from attrs-standard name",
        ),
        pytest.param(
            xr.DataArray(
                [0.1],
                coords={
                    "cell_ids": (
                        "cells",
                        [1],
                        {
                            "grid_name": "healpix",
                            "level": 1,
                            "indexing_scheme": "ring",
                        },
                    )
                },
                dims="cells",
            ),
            None,
            None,
            id="dataarray-from attrs-standard name",
        ),
        pytest.param(
            xr.Dataset(
                coords={
                    "zone_ids": (
                        "zones",
                        [1],
                        {
                            "grid_name": "healpix",
                            "level": 1,
                            "indexing_scheme": "ring",
                        },
                    )
                }
            ),
            None,
            "zone_ids",
            id="dataset-from attrs-custom name",
        ),
        pytest.param(
            xr.Dataset(coords={"cell_ids": ("cells", [1])}),
            {"grid_name": "healpix", "level": 1, "indexing_scheme": "ring"},
            None,
            id="dataset-dict-standard name",
        ),
    ),
)
def test_decode(obj, grid_info, name) -> None:
    kwargs = {}
    if name is not None:
        kwargs["name"] = name

    if isinstance(grid_info, dict):
        expected_grid_info = grid_info
    elif isinstance(grid_info, xdggs.DGGSInfo):
        expected_grid_info = grid_info.to_dict()
    else:
        expected_grid_info = obj[name if name is not None else "cell_ids"].attrs

    actual = obj.dggs.decode(grid_info, **kwargs)
    assert any(isinstance(index, xdggs.DGGSIndex) for index in actual.xindexes.values())
    assert actual.dggs.grid_info.to_dict() == expected_grid_info


@pytest.mark.parametrize(
    ["obj", "expected"],
    (
        (
            xr.DataArray(
                [0],
                coords={
                    "cell_ids": (
                        "cells",
                        [3],
                        {
                            "grid_name": "healpix",
                            "level": 1,
                            "indexing_scheme": "ring",
                        },
                    )
                },
                dims="cells",
            ),
            xr.Dataset(
                coords={
                    "latitude": ("cells", [66.44353569089877]),
                    "longitude": ("cells", [315.0]),
                }
            ),
        ),
        (
            xr.Dataset(
                coords={
                    "cell_ids": (
                        "cells",
                        [0x832830FFFFFFFFF],
                        {"grid_name": "h3", "level": 3},
                    )
                }
            ),
            xr.Dataset(
                coords={
                    "latitude": ("cells", [38.19320895]),
                    "longitude": ("cells", [-122.19619676]),
                }
            ),
        ),
    ),
)
def test_cell_centers(obj, expected):
    obj_ = obj.pipe(xdggs.decode)

    actual = obj_.dggs.cell_centers()

    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ["obj", "expected"],
    (
        (
            xr.DataArray(
                [0],
                coords={
                    "cell_ids": (
                        "cells",
                        [3],
                        {
                            "grid_name": "healpix",
                            "level": 1,
                            "indexing_scheme": "ring",
                        },
                    )
                },
                dims="cells",
            ),
            xr.DataArray(
                [0],
                coords={
                    "latitude": ("cells", [66.44353569089877]),
                    "longitude": ("cells", [315.0]),
                    "cell_ids": (
                        "cells",
                        [3],
                        {
                            "grid_name": "healpix",
                            "level": 1,
                            "indexing_scheme": "ring",
                        },
                    ),
                },
                dims="cells",
            ),
        ),
        (
            xr.Dataset(
                coords={
                    "cell_ids": (
                        "cells",
                        [0x832830FFFFFFFFF],
                        {"grid_name": "h3", "level": 3},
                    )
                }
            ),
            xr.Dataset(
                coords={
                    "latitude": ("cells", [38.19320895]),
                    "longitude": ("cells", [-122.19619676]),
                    "cell_ids": (
                        "cells",
                        [0x832830FFFFFFFFF],
                        {"grid_name": "h3", "level": 3},
                    ),
                }
            ),
        ),
    ),
)
def test_assign_latlon_coords(obj, expected):
    obj_ = obj.pipe(xdggs.decode)

    actual = obj_.dggs.assign_latlon_coords()

    xr.testing.assert_allclose(actual, expected)
