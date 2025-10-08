import numpy as np
import xarray as xr

from xdggs.conventions.registry import register_encoder
from xdggs.utils import GRID_REGISTRY, call_on_dataset


def infer_grid_name(index):
    for name, cls in GRID_REGISTRY.items():
        if cls is type(index):
            return name

    raise ValueError("unknown index")


@register_encoder("xdggs")
def xdggs(obj):
    def _convert(ds):
        coord = ds.dggs._name

        grid_name = infer_grid_name(ds.dggs.index)
        metadata = {"grid_name": grid_name} | ds.dggs.grid_info.to_dict()

        return ds.assign_coords({coord: lambda ds: ds[coord].assign_attrs(metadata)})

    return call_on_dataset(_convert, obj)


@register_encoder("easygems")
def easygems(obj):
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


@register_encoder("cf")
def cf(obj):
    def _convert(ds):
        grid_info = ds.dggs.grid_info
        dim = ds.dggs.index._dim
        coord = ds.dggs._name

        grid_name = infer_grid_name(ds.dggs.index)
        metadata = grid_info.to_dict() | {"grid_mapping_name": grid_name}
        metadata["refinement_level"] = metadata.pop("level")

        crs = xr.Variable((), np.int8(0), metadata)

        additional_var_attrs = {"coordinates": coord, "grid_mapping": "crs"}
        coord_attrs = {"standard_name": "healpix_index", "units": "1"}

        new = ds.copy(deep=False)
        for key, var in new.variables.items():
            if key == coord or dim not in var.dims:
                continue

            var.attrs |= additional_var_attrs

        new[coord].attrs |= coord_attrs

        return new.assign_coords({"crs": crs})

    return call_on_dataset(_convert, obj)
