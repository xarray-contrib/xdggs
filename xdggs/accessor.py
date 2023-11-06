import xarray as xr

from .index import DGGSIndex


@xr.register_dataset_accessor("dggs")
class DGGSAccessor:

    def __init__(self, obj):
        self._obj = obj
        
        indexes = {
            k: idx for k, idx in obj.xindexes.items() if isinstance(idx, DGGSIndex)
        }
        if len(indexes) > 1:
            raise ValueError("Only one DGGSIndex per object is supported")
        
        self._name, self._index = next(iter(indexes.items()))

    def sel_latlon(self, lat, lon):
        """Point-wise, nearest-neighbor selection from lat/lon data."""

        cell_indexers = {self._name: self._index._latlon2cellid(lat, lon)}
        return self._obj.sel(cell_indexers)

    def assign_latlon_coords(self):
        """Return a new object with latitude and longitude coordinates
        of the cell centers."""

        lat_data, lon_data = self._index.cell_centers
        return self._obj.assign_coords(
            latitude=(self._index._dim, lat_data),
            longitude=(self._index._dim, lon_data),
        )