import warnings

import geoarrow.pyarrow as ga
import shapely

from xdggs.tests.matchers import (  # noqa: F401
    Match,
    MatchResult,
    assert_exceptions_equal,
)

warnings.filterwarnings("ignore", message="numpy.ndarray size changed")


def geoarrow_to_shapely(arr):
    return shapely.from_wkb(ga.as_wkb(arr))
