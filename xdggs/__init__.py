from importlib.metadata import version

import xdggs.tutorial  # noqa: F401
from xdggs.accessor import DGGSAccessor  # noqa: F401
from xdggs.conventions import decode, encode
from xdggs.grid import DGGSInfo
from xdggs.h3 import H3Index, H3Info
from xdggs.healpix import HealpixIndex, HealpixInfo
from xdggs.index import DGGSIndex

__version__ = version("xdggs")

__all__ = [
    "__version__",
    "DGGSInfo",
    "H3Info",
    "HealpixInfo",
    "DGGSIndex",
    "H3Index",
    "HealpixIndex",
    "encode",
    "decode",
]
