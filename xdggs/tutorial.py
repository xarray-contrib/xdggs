from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import pooch
from xarray import open_dataset as _open_dataset

if TYPE_CHECKING:
    from xarray.backends.api import T_Engine

_default_cache_dir_name = "xdggs_tutorial_data"
base_url = "https://github.com/xdggs/xdggs-data"
version = "main"

external_urls = {}  # type: dict
file_formats = {
    "air_temperature": 4,
}


def _construct_cache_dir(path):
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    elif path is None:
        path = pooch.os_cache(_default_cache_dir_name)

    return path


def _check_netcdf_engine_installed(name):
    version = file_formats.get(name)
    if version == 3:
        try:
            import scipy  # noqa
        except ImportError:
            try:
                import netCDF4  # noqa
            except ImportError as err:
                raise ImportError(
                    f"opening tutorial dataset {name} requires either scipy or "
                    "netCDF4 to be installed."
                ) from err
    if version == 4:
        try:
            import h5netcdf  # noqa
        except ImportError:
            try:
                import netCDF4  # noqa
            except ImportError as err:
                raise ImportError(
                    f"opening tutorial dataset {name} requires either h5netcdf "
                    "or netCDF4 to be installed."
                ) from err


def open_dataset(
    name: str,
    grid_name: str,
    *,
    cache: bool = True,
    cache_dir: None | str | os.PathLike = None,
    engine: T_Engine = None,
    **kws,
):
    """
    Open a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Available datasets (available grid names in parentheses):

    * ``"air_temperature"`` (``h3``, ``healpix``): NCEP reanalysis subset.

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
        e.g. 'air_temperature'
    grid_name : str
        Name of the grid file.
    cache_dir : path-like, optional
        The directory in which to search for and write cached data.
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    **kws : dict, optional
        Passed to xarray.open_dataset

    See Also
    --------
    xarray.tutorial.open_dataset
    """
    import xdggs

    logger = pooch.get_logger()
    logger.setLevel("WARNING")

    cache_dir = _construct_cache_dir(cache_dir)
    if name in external_urls:
        url = external_urls[name]
    else:
        path = pathlib.Path(grid_name)
        if not path.suffix:
            # process the name
            default_extension = ".nc"
            if engine is None:
                _check_netcdf_engine_installed(grid_name)
            path = path.with_suffix(default_extension)

        url = f"{base_url}/raw/{version}/{name}/{path.name}"

    headers = {"User-Agent": f"xdggs/{xdggs.__version__}"}

    # retrieve the file
    downloader = pooch.HTTPDownloader(headers=headers)
    filepath = pooch.retrieve(
        url=url, known_hash=None, path=cache_dir, downloader=downloader
    )
    ds = _open_dataset(filepath, engine=engine, **kws)
    if not cache:
        ds = ds.load()
        pathlib.Path(filepath).unlink()

    return ds
