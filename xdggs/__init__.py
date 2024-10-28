from importlib.metadata import PackageNotFoundError, version

import xdggs.tutorial  # noqa: F401
from xdggs.accessor import DGGSAccessor  # noqa: F401
from xdggs.grid import DGGSInfo
from xdggs.h3 import H3Index, H3Info
from xdggs.healpix import HealpixIndex, HealpixInfo
from xdggs.index import DGGSIndex, decode

try:
    __version__ = version("xdggs")
except PackageNotFoundError:  # noqa # pragma: no cover
    # package is not installed
    __version__ = "9999"

__all__ = [
    "__version__",
    "DGGSInfo",
    "H3Info",
    "HealpixInfo",
    "DGGSIndex",
    "H3Index",
    "HealpixIndex",
    "decode",
]
