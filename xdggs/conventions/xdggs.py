import xarray as xr

from xdggs.conventions.registry import Convention, register_convention
from xdggs.conventions.utils import infer_grid_name
from xdggs.grid import DGGSInfo
from xdggs.utils import GRID_REGISTRY, call_on_dataset


@register_convention("xdggs")
class Xdggs(Convention):
    def decode(self, obj, grid_info, name, index_options):
        if name is None:
            name = "cell_ids"

        try:
            var = obj[name]
        except KeyError:
            raise ValueError("Cannot find the cell ids coordinate")

        if len(var.dims) != 1:
            # TODO: allow 0D
            raise ValueError("cell id coordinate must be 1D")
        [dim] = var.dims

        if grid_info is None:
            grid_info = var.attrs
        elif isinstance(grid_info, DGGSInfo):
            # TODO: avoid serializing / deserializing cycle
            grid_info = grid_info.to_dict()

        grid_name = grid_info["grid_name"]
        if grid_name not in GRID_REGISTRY:
            raise ValueError(f"unknown grid name: {grid_name}")
        index_cls = GRID_REGISTRY[grid_name]

        var_ = var.copy(deep=True)
        var_.attrs = grid_info
        index = index_cls.from_variables({name: var_}, options=index_options)

        return xr.Coordinates({name: var.variable}, indexes={name: index})

    def encode(self, obj):
        def _convert(ds):
            coord = ds.dggs._name

            grid_name = infer_grid_name(ds.dggs.index)
            metadata = {"grid_name": grid_name} | ds.dggs.grid_info.to_dict()

            return ds.assign_coords(
                {coord: lambda ds: ds[coord].assign_attrs(metadata)}
            )

        return call_on_dataset(_convert, obj)
