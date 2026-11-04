from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from xdggs import conventions
from xdggs.grid import DGGSInfo
from xdggs.index import DGGSIndex
from xdggs.plotting import explore

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lonboard.basemap import MaplibreBasemap
    from lonboard.experimental.view import BaseView
    from matplotlib.colors import Colormap

    from xdggs.plotting import MapWithControls


@xr.register_dataset_accessor("dggs")
@xr.register_dataarray_accessor("dggs")
class DGGSAccessor:
    _obj: xr.Dataset | xr.DataArray
    _index: DGGSIndex | None
    _name: str

    def __init__(self, obj: xr.Dataset | xr.DataArray):
        self._obj = obj

        index = None
        name = ""
        for k, idx in obj.xindexes.items():
            if isinstance(idx, DGGSIndex):
                if index is not None:
                    raise ValueError(
                        "Only one DGGSIndex per dataset or dataarray is supported"
                    )
                index = idx
                name = k
        self._name = name
        self._index = index

    def decode(
        self,
        grid_info: Mapping[str, object] | DGGSInfo | None = None,
        *,
        name: str = "cell_ids",
        convention: str | None = None,
        index_options: Mapping[str, Any] | None = None,
        **index_kwargs,
    ) -> xr.Dataset | xr.DataArray:
        """decode the DGGS cell ids

        Parameters
        ----------
        grid_info : dict or xdggs.DGGSInfo, optional
            Override the grid parameters on the dataset. Useful to set attributes on
            the dataset.
        name : str, optional
            The name of the coordinate containing the cell ids. The default name
            depends on the convention.
        convention : str, optional
            The name of the metadata convention. Built-in conventions are:

            - "zarr": the zarr dggs convention. All metadata is contained in a
              single nested JSON object that lives in the group attributes.
            - "cf": the DGGS section from the CF conventions. While the
              convention extension is specialized on ``healpix`` for now, the
              decoder can work with other DGGS as well. For this, all metadata
              lives on a variable with a ``grid_mapping_name`` attribute, and
              the cell ids coordinate is indicated by the ``coordinates``
              attribute on data variables / other coordinates (this can be
              overridden by the ``name`` parameter).
            - "xdggs": the existing xdggs convention. ``name`` points to the
              coordinate containing cell ids, and which has all the grid
              metadata. The ``name`` parameter defaults to ``"cell_ids"``.

            If not passed, will try to infer the convention in the order listed here.
        index_options, **index_kwargs : dict, optional
            Additional options to forward to the index.

        Returns
        -------
        obj : xarray.DataArray or xarray.Dataset
            The object with a DGGS index on the cell id coordinate.

        Examples
        --------
        >>> import xdggs
        >>> import xarray as xr
        >>> import numpy as np

        Dataset following the xdggs convention:

        >>> ds = xr.Dataset(
        ...     {"a": ("cells", list("abcdefghijkl"))},
        ...     coords={
        ...         "cell_ids": (
        ...             "cells",
        ...             np.arange(12, dtype="uint64"),
        ...             {
        ...                 "grid_name": "healpix",
        ...                 "level": 0,
        ...                 "indexing_scheme": "nested",
        ...             },
        ...         )
        ...     },
        ... )
        >>> ds.dggs.decode()
        <xarray.Dataset> Size: 144B
        Dimensions:   (cells: 12)
        Coordinates:
          * cell_ids  (cells) uint64 96B 0 1 2 3 4 5 6 7 8 9 10 11
        Dimensions without coordinates: cells
        Data variables:
            a         (cells) <U1 48B 'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j' 'k' 'l'
        Indexes:
            cell_ids  HealpixIndex(level=0, indexing_scheme=nested, kind=pandas)
        """
        return conventions.decode(
            self._obj,
            grid_info=grid_info,
            name=name,
            convention=convention,
            index_options=index_options,
            **index_kwargs,
        )

    @property
    def index(self) -> DGGSIndex:
        """The DGGSIndex instance for this Dataset or DataArray.

        Raises
        ------
        ValueError
            if no DGGSIndex can be found
        """
        if self._index is None:
            raise ValueError("no DGGSIndex found on this Dataset or DataArray")
        return self._index

    @property
    def coord(self) -> xr.DataArray:
        """The indexed DGGS (cell ids) coordinate as a DataArray.

        Raises
        ------
        ValueError
            if no such coordinate is found on the Dataset / DataArray
        """
        if not self._name:
            raise ValueError(
                "no coordinate with a DGGSIndex found on this Dataset or DataArray"
            )
        return self._obj[self._name]

    @property
    def params(self) -> dict:
        """The grid parameters after normalization."""
        return self.index.grid.to_dict()

    @property
    def grid_info(self) -> DGGSInfo:
        """The grid info object containing the DGGS type and its parameters.

        Returns
        -------
        xdggs.DGGSInfo
        """
        return self.index.grid_info

    def sel_latlon(
        self, latitude: npt.ArrayLike, longitude: npt.ArrayLike
    ) -> xr.Dataset | xr.DataArray:
        """Select grid cells from latitude/longitude data.

        Parameters
        ----------
        latitude : array-like
            Latitude coordinates (degrees).
        longitude : array-like
            Longitude coordinates (degrees).

        Returns
        -------
        subset : xarray.Dataset or xarray.DataArray
            A new :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray`
            with all cells that contain the input latitude/longitude data points.
        """
        cell_indexers = {
            self._name: self.grid_info.geographic2cell_ids(lon=longitude, lat=latitude)
        }
        return self._obj.sel(cell_indexers)

    def assign_latlon_coords(self) -> xr.Dataset | xr.DataArray:
        """Return a new Dataset or DataArray with new "latitude" and "longitude"
        coordinates representing the grid cell centers."""

        lon_data, lat_data = self.index.cell_centers()

        return self._obj.assign_coords(
            latitude=(self.index._dim, lat_data),
            longitude=(self.index._dim, lon_data),
        )

    @property
    def cell_ids(self):
        """The indexed DGGS (cell ids) coordinate as a DataArray.

        Alias of ``coord``.

        Raises
        ------
        ValueError
            if no such coordinate is found on the Dataset / DataArray
        """
        return self.coord

    def cell_centers(self):
        """derive geographic cell center coordinates

        Returns
        -------
        coords : xarray.Dataset
            Dataset containing the cell centers in geographic coordinates.
        """
        lon_data, lat_data = self.index.cell_centers()

        return xr.Dataset(
            coords={
                "latitude": (self.index._dim, lat_data),
                "longitude": (self.index._dim, lon_data),
            }
        )

    def cell_boundaries(self):
        """derive cell boundary polygons

        Returns
        -------
        boundaries : xarray.DataArray
            The cell boundaries as shapely objects.
        """
        boundaries = self.index.cell_boundaries()

        return xr.DataArray(
            boundaries, coords={self._name: self.cell_ids}, dims=self.cell_ids.dims
        )

    def zoom_to(self, level: int):
        """Change the refinement level of the cell ids to `level`.

        Parameters
        ----------
        level : int
            The refinement level to change to. Can be smaller than the dataset's
            level to compute parents, or bigger to fetch the children. In the
            latter case, the array will have an additional `"children"`
            dimension.

        Returns
        -------
        zoomed : xr.DataArray
            The children or parents of the current cells.
        """
        zoomed = self.index.zoom_to(level=level)

        if zoomed.ndim == 1:
            dims = self.cell_ids.dims
        else:
            dims = [*self.cell_ids.dims, "children"]

        return xr.DataArray(zoomed, coords={self._name: self.cell_ids}, dims=dims)

    def explore(
        self,
        *,
        cmap: str | Colormap = "viridis",
        center: int | float | None = None,
        vmin: int | float | None = None,
        vmax: int | float | None = None,
        robust: bool = False,
        alpha: float | None = None,
        coords: list[str] | None = None,
        view: BaseView | None = None,
        basemap: MaplibreBasemap | None = None,
    ) -> MapWithControls:
        """interactively explore the data using `lonboard`

        Requires `lonboard`, `matplotlib`, and `arro3.core` to be installed.

        Parameters
        ----------
        cmap : str
            The name of the color map to use
        center : int or float, optional
            If set, will use this as the center value of a diverging color map.
        alpha : float, optional
            If set, controls the transparency of the polygons.
        coords : list of str, optional
            Additional coordinates to contain in the table of
            contents. Defaults to ``["latitude", "longitude"]``.

        Returns
        -------
        map : lonboard.Map
            The rendered map.

        Notes
        -----
        Plotting currently is restricted to 1D `DataArray` objects.
        """
        return explore(
            self._obj,
            colorize_params={
                "cmap": cmap,
                "center": center,
                "vmin": vmin,
                "vmax": vmax,
                "robust": robust,
                "alpha": alpha,
            },
            coords=coords,
            view=view,
            basemap=basemap,
        )

    def to_raster(
        self,
        sampling_grid: Any = "native",
        *,
        interpolation: str = "nearest",
    ) -> xr.Dataset | xr.DataArray:
        """Resample the data onto a georeferenced raster.

        Requires a rendering backend for the grid: ``healpix-plot`` for
        HEALPix. Other grids are not supported yet. The result can be written
        to a GeoTIFF with ``rioxarray`` (``result.rio.to_raster(path)``) and
        converted back with :py:func:`xdggs.from_raster`.

        Parameters
        ----------
        sampling_grid : str, dict or sampling grid object, default: "native"
            The target pixel grid:

            - ``"native"``: a grid in the grid's native projection with exactly
              one pixel per cell, covering the bounding box of the cells. The
              conversion is lossless. For HEALPix, the pixels are squares
              rotated by 45° in ``+proj=healpix`` (requires ``level >= 1``).
            - dict: parameters of a regular longitude / latitude grid, passed to
              the backend (for HEALPix, ``shape``, ``resolution`` and
              ``center``; see ``healpix_plot.SamplingGrid``).
            - a backend sampling grid object, for example
              ``healpix_plot.AffineSamplingGrid`` with a ``crs``.
        interpolation : {"nearest"}, default: "nearest"
            How to fill the pixels. Each pixel takes the value of the cell
            containing its centre.

        Returns
        -------
        raster : xarray.DataArray or xarray.Dataset
            The data with the cell dimension replaced by ``y`` and ``x``. Pixels
            whose cell is missing from the data are set to ``NaN``. If
            ``rioxarray`` is installed, the CRS and the affine transform are
            written with the ``rio`` accessor, otherwise they are stored in
            the ``crs_wkt`` and ``transform`` attributes. Coordinates along
            the cell dimension are dropped.

        See Also
        --------
        xdggs.from_raster
        """
        from xdggs.rendering import to_raster

        return to_raster(
            self._obj, sampling_grid=sampling_grid, interpolation=interpolation
        )

    def plot(
        self,
        *,
        projection: Any = "Mollweide",
        sampling_grid: Any = None,
        ax: Any = None,
        **kwargs,
    ):
        """Plot the data on a static map using matplotlib and cartopy.

        Requires a rendering backend for the grid: ``healpix-plot`` for
        HEALPix. Other grids are not supported yet.

        Parameters
        ----------
        projection : str or cartopy.crs.Projection, default: "Mollweide"
            The map projection. Strings are names of cartopy projections, or
            ``"HEALPix"`` for the HEALPix projection on the grid's ellipsoid.
            Ignored if ``ax`` is given.
        sampling_grid : str, dict or sampling grid object, optional
            The grid the data is resampled to before drawing; see
            :py:meth:`DataArray.dggs.to_raster`. Defaults to ``"native"`` (the
            cells themselves) for the HEALPix projection, and a longitude /
            latitude image of 1024 x 1024 pixels otherwise.
        ax : cartopy.mpl.geoaxes.GeoAxes, optional
            The axes to draw on. If not given, a new figure is created.
        **kwargs
            Passed on to the backend, for HEALPix ``healpix_plot.plot``
            (``cmap``, ``vmin``, ``vmax``, ``norm``, ``colorbar``, ``title``,
            ``view``, ...).

        Returns
        -------
        mappable : matplotlib.cm.ScalarMappable
            The image or mesh drawn on the map.

        Notes
        -----
        Plotting currently is restricted to 1D `DataArray` objects.
        """
        from xdggs.rendering import get_rendering_backend

        if not isinstance(self._obj, xr.DataArray):
            raise TypeError(
                "plotting is only supported for DataArray objects, select a variable first"
            )
        if self._obj.ndim != 1:
            raise ValueError(
                "plotting is only supported for 1D data, select along"
                f" {[dim for dim in self._obj.dims if dim not in self.coord.dims]} first"
            )

        grid_info = self.grid_info
        return get_rendering_backend(grid_info).plot(
            grid_info,
            np.asarray(self.coord.data),
            np.asarray(self._obj.data),
            projection=projection,
            sampling_grid=sampling_grid,
            ax=ax,
            **kwargs,
        )

    def encode(self, convention: str, *, encoding: dict[str, Any] | None = None):
        """Encode the dataset to the given convention

        Parameters
        ----------
        convention : str
            The name of the convention. Supported are:

            - "easygems": ``grid_mapping`` coordinate and ``cell`` dimension and ``cell`` coordinate with a `pandas` index.
            - "cf": ``grid_mapping`` coordinate with ``cell_index`` coordinate and ``cell`` dimension.
            - "xdggs": ``cell_ids`` coordinate with grid metadata and a ``cells`` coordinate.

        Returns
        -------
        obj : xarray.DataArray or xarray.Dataset
            The object converted to the given dimension.
        """
        return conventions.encode(self._obj, convention=convention, encoding=encoding)
