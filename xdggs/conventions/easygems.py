import numpy as np
import xarray as xr

from xdggs.conventions.base import Convention
from xdggs.conventions.registry import register_convention
from xdggs.utils import call_on_dataset


@register_convention("easygems")
class Easygems(Convention):
    def encode(self, obj):
        orders = {"nested": "nest", "ring": "ring"}

        def _convert(ds):
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

            return (
                ds.assign_coords(crs=crs)
                .drop_indexes(coord)
                .rename_dims({dim: "cell"})
                .rename_vars({coord: "cell"})
                .set_xindex("cell")
            )

        return call_on_dataset(_convert, obj)
