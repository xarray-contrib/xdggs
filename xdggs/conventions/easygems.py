import numpy as np
import xarray as xr

from xdggs.conventions.base import Convention
from xdggs.conventions.registry import register_convention


@register_convention("easygems")
class Easygems(Convention):
    def encode(self, ds: xr.Dataset, *, encoding: None = None) -> xr.Dataset:
        orders = {"nested": "nest", "ring": "ring"}

        grid_info = ds.dggs.grid_info
        dim = ds.dggs.index._dim
        coord = ds.dggs._name

        order = orders.get(grid_info.indexing_scheme)
        if order is None:
            raise ValueError(f"easygems: unsupported indexing scheme: {order}")

        metadata = {
            "grid_mapping_name": "healpix",
            "healpix_nside": grid_info.nside,
            "healpix_order": order,
        }
        crs = xr.Variable((), np.int8(0), metadata)
        crs_name = encoding.get("grid_mapping_variable", "crs")

        return (
            ds.assign_coords({crs_name: crs})
            .drop_indexes(coord)
            .rename_dims({dim: "cell"})
            .rename_vars({coord: "cell"})
        )
