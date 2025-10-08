import xarray as xr

from xdggs.conventions.registry import register_decoder
from xdggs.grid import DGGSInfo
from xdggs.utils import GRID_REGISTRY


@register_decoder("xdggs")
def xdggs(obj, grid_info, name):
    try:
        var = obj[name]
    except IndexError:
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

    index = index_cls.from_variables({name: var}, options=grid_info)

    return xr.Coordinates({name: var.variable}, indexes={name: index})
