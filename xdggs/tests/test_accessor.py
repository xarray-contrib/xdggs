import pytest
import xarray as xr

import xdggs


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
                            "resolution": 1,
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
                        {"grid_name": "h3", "resolution": 3},
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
                            "resolution": 1,
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
                            "resolution": 1,
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
                        {"grid_name": "h3", "resolution": 3},
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
                        {"grid_name": "h3", "resolution": 3},
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
