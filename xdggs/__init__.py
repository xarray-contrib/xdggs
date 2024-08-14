from importlib.metadata import PackageNotFoundError, version

from xdggs.accessor import DGGSAccessor  # noqa: F401
from xdggs.grid import DGGSInfo
from xdggs.h3 import H3Index
from xdggs.healpix import HealpixIndex
from xdggs.index import DGGSIndex, decode

try:
    __version__ = version("xdggs")
except PackageNotFoundError:  # noqa # pragma: no cover
    # package is not installed
    __version__ = "9999"

__all__ = ["__version__", "DGGSIndex", "H3Index", "HealpixIndex", "decode", "DGGSInfo"]
