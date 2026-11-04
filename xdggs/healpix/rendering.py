"""Rendering backend for HEALPix grids, built on ``healpix-plot``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from xdggs.healpix.grid_info import HealpixInfo, _serialize_ellipsoid
from xdggs.rendering import register_rendering_backend

if TYPE_CHECKING:
    import xarray as xr


def _import_healpix_plot():
    try:
        import healpix_plot
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "rasterising and plotting HEALPix data requires `healpix-plot`."
            " Install it with `pip install 'xdggs[plot]'`."
        ) from e

    return healpix_plot


def healpix_grid(grid_info: HealpixInfo):
    """The :py:class:`healpix_plot.HealpixGrid` equivalent of ``grid_info``."""
    healpix_plot = _import_healpix_plot()

    if grid_info.level is None:
        raise ValueError("rendering requires cells of a single level")

    return healpix_plot.HealpixGrid(
        level=grid_info.level,
        indexing_scheme=grid_info.indexing_scheme,
        ellipsoid=_serialize_ellipsoid(grid_info.ellipsoid),
    )


def resolve_sampling_grid(sampling_grid: Any, grid, cell_ids: np.ndarray):
    from healpix_plot.sampling_grid import (
        AffineSamplingGrid,
        ParametrizedSamplingGrid,
    )

    if isinstance(sampling_grid, str):
        if sampling_grid != "native":
            raise ValueError(f"unknown sampling grid: {sampling_grid!r}")
        return AffineSamplingGrid.from_healpix(grid, cell_ids)
    elif isinstance(sampling_grid, dict):
        return ParametrizedSamplingGrid.from_dict(sampling_grid)

    return sampling_grid


def _is_healpix_projection(projection: Any) -> bool:
    if isinstance(projection, str):
        return projection.lower() == "healpix"

    healpix_plot = _import_healpix_plot()
    return isinstance(projection, healpix_plot.HEALPix)


class HealpixPlotBackend:
    def sample(
        self, grid_info: HealpixInfo, cell_ids: np.ndarray, sampling_grid: Any
    ) -> tuple[np.ma.MaskedArray, xr.DataArray]:
        from healpix_plot.raster import to_dataarray

        grid = healpix_grid(grid_info)
        target_grid = resolve_sampling_grid(sampling_grid, grid, cell_ids).resolve(
            cell_ids, grid
        )
        pixel_cell_ids = target_grid.cell_ids(grid)
        template = to_dataarray(
            np.zeros(target_grid.shape, dtype=bool), target_grid, grid
        )

        return pixel_cell_ids, template

    def raster_cell_ids(
        self, grid_info: HealpixInfo, raster: xr.DataArray | xr.Dataset
    ) -> np.ma.MaskedArray:
        from healpix_plot.raster import raster_cell_ids

        return raster_cell_ids(raster, healpix_grid(grid_info))

    def plot(
        self,
        grid_info: HealpixInfo,
        cell_ids: np.ndarray,
        data: np.ndarray,
        *,
        sampling_grid: Any = None,
        projection: Any = "Mollweide",
        **kwargs,
    ):
        healpix_plot = _import_healpix_plot()

        grid = healpix_grid(grid_info)
        if sampling_grid is None:
            # exact cells on the HEALPix plane, a lat / lon image otherwise
            sampling_grid = (
                "native" if _is_healpix_projection(projection) else {"shape": 1024}
            )

        return healpix_plot.plot(
            cell_ids,
            data,
            healpix_grid=grid,
            sampling_grid=resolve_sampling_grid(sampling_grid, grid, cell_ids),
            projection=projection,
            **kwargs,
        )


@register_rendering_backend(HealpixInfo)
def _healpix_plot_backend() -> HealpixPlotBackend:
    _import_healpix_plot()
    return HealpixPlotBackend()
