from importlib.metadata import PackageNotFoundError, version

from xdggs.accessor import DGGSAccessor  # noqa
from xdggs.healpix import HealpixIndex
from xdggs.index import DGGSIndex

try:
    __version__ = version("xdggs")
except PackageNotFoundError:  # noqa
    # package is not installed
    pass

__all__ = ["__version__", "DGGSIndex", "HealpixIndex"]
