from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import xarray as xr

from xdggs.grid import DGGSInfo
from xdggs.healpix import HealpixInfo
from xdggs.healpix import downscale as healpix_downscale
from xdggs.index import DGGSIndex
from xdggs.plotting import explore


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

    def decode(self, grid_info=None, *, name="cell_ids") -> xr.Dataset | xr.DataArray:
        """decode the DGGS cell ids

        Parameters
        ----------
        grid_info : dict or DGGSInfo, optional
            Override the grid parameters on the dataset. Useful to set attributes on
            the dataset.
        name : str, default: "cell_ids"
            The name of the coordinate containing the cell ids.

        Returns
        -------
        obj : xarray.DataArray or xarray.Dataset
            The object with a DGGS index on the cell id coordinate.
        """
        var = self._obj[name]
        if isinstance(grid_info, DGGSInfo):
            grid_info = grid_info.to_dict()
        if isinstance(grid_info, dict):
            var.attrs = grid_info

        return self._obj.drop_indexes(name, errors="ignore").set_xindex(name, DGGSIndex)

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
        subset
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

    def explore(self, *, cmap="viridis", center=None, alpha=None, coords=None):
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
        coords : list of str, default: ["latitude", "longitude"]
            Additional coordinates to contain in the table of contents.

        Returns
        -------
        map : lonboard.Map
            The rendered map.

        Notes
        -----
        Plotting currently is restricted to 1D `DataArray` objects.
        """
        if isinstance(self._obj, xr.Dataset):
            raise ValueError("does not work with Dataset objects, yet")

        return explore(
            self._obj,
            cmap=cmap,
            center=center,
            alpha=alpha,
            coords=coords,
        )

    def downscale(self, level: int, agg: Callable = None):
        """Aggregate data to a lower grid level.

        Parameters
        ----------
        level : int, optional
            The target level of the grid you want to group towards. This is the level of the resulting data.
        agg : callable, default: np.mean
            The aggregation function to use. This function must accept a 1D array and return a scalar value.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The downscaled data.
        """
        if agg is None:
            agg = np.mean

        assert_valid_level(level)

        if self.grid_info.level < level:
            raise ValueError(
                f"Can't downscale to level {level} from data on level {self.grid_info.level}. Did you mean upscale?"
            )

        offset = self.grid_info.level - level

        if not isinstance(self.grid_info, HealpixInfo):
            raise ValueError(
                "Downscaling is currently only supported for Healpix grids."
            )

        return healpix_downscale(
            self._obj, offset=offset, agg=agg, grid_info=self.grid_info
        )

    def upscale(self, level: int):
        if not isinstance(level, int):
            raise ValueError(
                f"Expected level to be of type {{int}}. Got {type(level).__name__}"
            )

        if self.grid_info.level > level:
            raise ValueError(
                f"Can't upscale to level {level} from data on level {self.grid_info.level}. Did you mean downscale?"
            )

        offset = level - self.grid_info.level  # noqa

        if not isinstance(self.grid_info, HealpixInfo):
            raise ValueError("Upscaling is currently only supported for Healpix grids.")

        raise NotImplementedError()

    def rescale(self, level: int, downscale_agg: Callable | None = None):
        """Rescale the data to a different grid level by either upscaling or downscaling.

        Parameters
        ----------
        level : int
            The target level of the grid you want to group towards. This is the level of the resulting data.
        downscale_agg : callable, default: np.mean
            The aggregation function to use if downscaling. This function must accept a 1D array and return a scalar value.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The rescaled data.
        """
        assert_valid_level(level)

        if self.grid_info.level < level:
            return self.upscale(level)
        else:
            return self.downscale(level, agg=downscale_agg)


def assert_valid_level(level: int) -> None:
    if not isinstance(level, int):
        raise ValueError(f"level must be an integer, got {type(level).__name__}")

    if level < 0:
        raise ValueError(f"level must be a non-negative integer, got {level}")
