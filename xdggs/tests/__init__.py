from contextlib import nullcontext

import geoarrow.pyarrow as ga
import pytest
import shapely

from xdggs.tests.matchers import (  # noqa: F401
    Match,
    MatchResult,
    assert_exceptions_equal,
)

try:
    import dask
    import dask.array as da

    has_dask = True
except ImportError:
    dask = None

    class da:
        @staticmethod
        def arange(*args, **kwargs):
            pass

    has_dask = False


# vendored from xarray
class CountingScheduler:
    """Simple dask scheduler counting the number of computes.

    Reference: https://stackoverflow.com/questions/53289286/"""

    def __init__(self, max_computes=0):
        self.total_computes = 0
        self.max_computes = max_computes

    def __call__(self, dsk, keys, **kwargs):
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            raise RuntimeError(
                f"Too many computes. Total: {self.total_computes} > max: {self.max_computes}."
            )
        return dask.get(dsk, keys, **kwargs)


requires_dask = pytest.mark.skipif(not has_dask, reason="requires dask")


def geoarrow_to_shapely(arr):
    return shapely.from_wkb(ga.as_wkb(arr))


# vendored from xarray
def raise_if_dask_computes(max_computes=0):
    # return a dummy context manager so that this can be used for non-dask objects
    if not has_dask:
        return nullcontext()
    scheduler = CountingScheduler(max_computes)
    return dask.config.set(scheduler=scheduler)
