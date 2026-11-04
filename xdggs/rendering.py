"""Static rendering: rasterisation and matplotlib / cartopy plots.

The grid geometry (which cell lies under each pixel, how to draw the cells) is
delegated to a backend chosen by the type of the grid info object. Backends
are optional dependencies; HEALPix grids use ``healpix-plot``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
import xarray as xr

from xdggs.grid import DGGSInfo

if TYPE_CHECKING:
    from collections.abc import Callable


class RenderingBackend(Protocol):
    def sample(
        self, grid_info: DGGSInfo, cell_ids: np.ndarray, sampling_grid: Any
    ) -> tuple[np.ma.MaskedArray, xr.DataArray]:
        """The cell id under each pixel centre of ``sampling_grid``.

        Returns the cell ids as a masked ``(y, x)`` array (masked: pixel off the
        globe) and an empty, georeferenced ``(y, x)`` template carrying the
        spatial coordinates, the CRS and the transform.
        """

    def raster_cell_ids(
        self, grid_info: DGGSInfo, raster: xr.DataArray | xr.Dataset
    ) -> np.ma.MaskedArray:
        """The cell id under each pixel centre of an existing raster."""

    def plot(
        self, grid_info: DGGSInfo, cell_ids: np.ndarray, data: np.ndarray, **kwargs
    ) -> Any:
        """Plot the data with matplotlib and return the mappable."""


_backends: dict[type[DGGSInfo], Callable[[], RenderingBackend]] = {}


def register_rendering_backend(grid_info_type: type[DGGSInfo]):
    """Register a rendering backend factory for a grid info type.

    The factory is called on use, so that the backend can import its
    (optional) dependencies lazily.
    """

    def register(factory: Callable[[], RenderingBackend]):
        _backends[grid_info_type] = factory
        return factory

    return register


def get_rendering_backend(grid_info: DGGSInfo) -> RenderingBackend:
    for cls in type(grid_info).__mro__:
        factory = _backends.get(cls)
        if factory is not None:
            return factory()

    raise NotImplementedError(
        f"rasterising and static plots are not supported for {type(grid_info).__name__}"
    )


def _spatial_variables(obj: xr.Dataset | xr.DataArray, dim: str) -> list[str]:
    return [name for name, coord in obj.coords.items() if dim in coord.dims]


def to_raster(
    obj: xr.Dataset | xr.DataArray,
    *,
    sampling_grid: Any,
    interpolation: str = "nearest",
) -> xr.Dataset | xr.DataArray:
    if interpolation != "nearest":
        raise ValueError(
            f"unsupported interpolation method {interpolation!r}, only 'nearest' is"
            " available"
        )

    coord = obj.dggs.coord
    [dim] = coord.dims
    if {"x", "y"} & set(obj.dims):
        raise ValueError("the object must not have dimensions named 'x' or 'y'")

    cell_ids = np.asarray(coord.data)
    grid_info = obj.dggs.grid_info
    backend = get_rendering_backend(grid_info)

    pixel_cell_ids, template = backend.sample(grid_info, cell_ids, sampling_grid)

    # position of each pixel's cell in `obj`, or -1 if the cell is not in it
    positions = (
        pd.Index(cell_ids)
        .get_indexer(np.ma.getdata(pixel_cell_ids).ravel())
        .reshape(pixel_cell_ids.shape)
    )
    valid = (positions != -1) & ~np.ma.getmaskarray(pixel_cell_ids)

    indexer = xr.Variable(template.dims, np.where(valid, positions, 0))
    mask = xr.DataArray(valid, dims=template.dims)

    def mask_pixels(arr):
        return arr.where(mask) if set(mask.dims) <= set(arr.dims) else arr

    stripped = obj.drop_vars(_spatial_variables(obj, dim))
    if isinstance(stripped, xr.DataArray):
        raster = stripped.isel({dim: indexer}).where(mask)
    else:
        raster = stripped.isel({dim: indexer}).map(mask_pixels)

    raster = raster.assign_coords(template.coords)
    raster.attrs = template.attrs | raster.attrs
    if "grid_mapping" in template.encoding:
        variables = [raster] if isinstance(raster, xr.DataArray) else raster.values()
        for variable in variables:
            variable.encoding["grid_mapping"] = template.encoding["grid_mapping"]

    return raster


def from_raster(
    raster: xr.Dataset | xr.DataArray,
    grid_info: DGGSInfo,
    *,
    name: str = "cell_ids",
    dim: str = "cells",
) -> xr.Dataset | xr.DataArray:
    """Convert a georeferenced raster to DGGS cells.

    Each pixel is assigned to the cell containing its centre. Pixels off the
    globe are dropped. This is the inverse of
    :py:meth:`DataArray.dggs.to_raster`: a raster written on the ``"native"``
    sampling grid is converted back to the original cells (plus the pixels
    of the bounding box that are not in the data, as ``NaN``).

    Requires ``rioxarray`` and a rendering backend for the grid
    (``healpix-plot`` for HEALPix).

    Parameters
    ----------
    raster : xarray.DataArray or xarray.Dataset
        The raster, with the ``rio`` accessor (for example opened with
        :py:func:`rioxarray.open_rasterio`).
    grid_info : xdggs.DGGSInfo
        The grid to convert to.
    name : str, default: "cell_ids"
        The name of the cell ids coordinate.
    dim : str, default: "cells"
        The name of the cell dimension.

    Returns
    -------
    obj : xarray.DataArray or xarray.Dataset
        The data with the ``y`` and ``x`` dimensions replaced by ``dim`` and a
        DGGS index on ``name``.

    Raises
    ------
    ValueError
        If the pixels are smaller than the cells, i.e. some cells contain more
        than one pixel centre.
    """
    from xdggs.conventions import decode

    backend = get_rendering_backend(grid_info)
    pixel_cell_ids = backend.raster_cell_ids(grid_info, raster)

    y_dim, x_dim = raster.rio.y_dim, raster.rio.x_dim
    on_globe = ~np.ma.getmaskarray(pixel_cell_ids).ravel()
    cell_ids = np.ma.getdata(pixel_cell_ids).ravel()[on_globe]

    if np.unique(cell_ids).size != cell_ids.size:
        raise ValueError(
            "some cells contain more than one pixel centre; only rasters with"
            " pixels at least as large as the cells can be converted"
        )

    spatial = [
        coord_name
        for coord_name, coord in raster.coords.items()
        if {y_dim, x_dim} & set(coord.dims) or coord_name == "spatial_ref"
    ]
    stacked = (
        raster.drop_vars(spatial)
        .stack({dim: [y_dim, x_dim]}, create_index=False)
        .isel({dim: np.flatnonzero(on_globe)})
        .assign_coords({name: (dim, cell_ids)})
    )
    # georeferencing metadata written by `to_raster`
    stacked.attrs = {
        key: value
        for key, value in stacked.attrs.items()
        if not key.startswith("healpix_") and key not in {"crs_wkt", "transform"}
    }
    stacked.encoding.pop("grid_mapping", None)

    return decode(stacked, grid_info, name=name, convention="xdggs")
