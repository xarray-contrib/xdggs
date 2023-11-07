from importlib.metadata import PackageNotFoundError, version

from xdggs.accessor import DGGSAccessor  # noqa
from xdggs.h3 import H3Index
from xdggs.healpix import HealpixIndex
from xdggs.index import DGGSIndex

try:
    __version__ = version("xdggs")
except PackageNotFoundError:  # noqa
    # package is not installed
    __version__ = "9999"

__all__ = ["__version__", "DGGSIndex", "H3Index", "HealpixIndex"]
