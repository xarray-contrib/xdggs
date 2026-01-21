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


def diff_dict_keys(left, right):
    left_only = left.keys() - right.keys()
    right_only = right.keys() - left.keys()

    diff = []
    if left_only:
        diff.append("- only in left:")
        diff.extend([f"   {k}: {v}" for k, v in left.items() if k in left_only])
    if right_only:
        diff.append("- only in right:")
        diff.extend([f"   {k}: {v}" for k, v in right.items() if k in right_only])

    return "\n".join(["mismatch in indexes:", *diff])


def diff_indexes(left, right, mismatching):
    diffs = [
        "\n".join(
            [
                f"Indexed variable: {name}",
                f"L  {left[name]}",
                f"R  {right[name]}",
            ]
        )
        for name in mismatching
    ]

    return "\n".join(["indexes do not match", "", *diffs])


def assert_indexes_equal(left, right):
    __tracebackhide__ = True

    assert left.keys() == right.keys(), diff_dict_keys(left, right)

    mismatching = [
        k
        for k in left
        if type(left[k]) is not type(right[k]) or not left[k].equals(right[k])
    ]

    assert not mismatching, diff_indexes(left, right, mismatching)
