from importlib.metadata import PackageNotFoundError, version

from .accessor import DGGSAccessor   # noqa
from .index import DGGSIndex
from .healpix import HealpixIndex


try:
    __version__ = version("xdggs")
except PackageNotFoundError:  # noqa
    # package is not installed
    pass

__all__ = ["__version__", "DGGSIndex", "HealpixIndex"]
