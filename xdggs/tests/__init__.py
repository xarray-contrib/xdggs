import geoarrow.pyarrow as ga
import shapely

from xdggs.tests.matchers import (  # noqa: F401
    Match,
    MatchResult,
    assert_exceptions_equal,
)


def geoarrow_to_shapely(arr):
    return shapely.from_wkb(ga.as_wkb(arr))
