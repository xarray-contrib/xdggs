import numpy.typing as npt
import xarray as xr

from xdggs.grid import DGGSInfo
from xdggs.index import DGGSIndex


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
            Override the grid information.
        name : str, default: "cell_ids"
            The name of the coordinate containing the cell ids.

        Returns
        -------
        obj : xarray.DataArray or xarray.Dataset
            The object with a new index.
        """
        var = self._obj[name]
        if isinstance(grid_info, DGGSInfo):
            grid_info = grid_info.to_dict()
        if isinstance(grid_info, dict):
            var.attrs = grid_info

        return self._obj.drop_indexes(name, errors="ignore").set_xindex(name, DGGSIndex)

    @property
    def index(self) -> DGGSIndex:
        """Returns the DGGSIndex instance for this Dataset or DataArray.

        Raise a ``ValueError`` if no such index is found.
        """
        if self._index is None:
            raise ValueError("no DGGSIndex found on this Dataset or DataArray")
        return self._index

    @property
    def coord(self) -> xr.DataArray:
        """Returns the indexed DGGS (cell ids) coordinate as a DataArray.

        Raise a ``ValueError`` if no such coordinate is found on this Dataset or DataArray.

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
            self._name: self.grid_info.geographic2cell_ids(latitude, longitude)
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

    def cell_centers(self):
        lon_data, lat_data = self.index.cell_centers()

        return xr.Dataset(
            coords={
                "latitude": (self.index._dim, lat_data),
                "longitude": (self.index._dim, lon_data),
            }
        )
