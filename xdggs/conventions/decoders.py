import xarray as xr

from xdggs.conventions.registry import register_decoder
from xdggs.grid import DGGSInfo
from xdggs.utils import GRID_REGISTRY, call_on_dataset


@register_decoder("xdggs")
def xdggs(obj, grid_info, name, index_options):
    if name is None:
        name = "cell_ids"

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

    var_ = var.copy(deep=True)
    var_.attrs = grid_info
    index = index_cls.from_variables({name: var_}, options=index_options)

    return xr.Coordinates({name: var.variable}, indexes={name: index})


@register_decoder("cf")
def cf(obj, grid_info, name, index_options):
    vars_ = call_on_dataset(
        lambda ds: ds.variables,
        obj,
    )
    grid_mapping_vars = {
        name: var for name, var in vars_.items() if "grid_mapping_name" in var.attrs
    }
    if len(grid_mapping_vars) != 1:
        raise ValueError("needs exactly one grid mapping variable for now")
    crs = next(iter(grid_mapping_vars.values()))

    if name is None:
        standard_name = f"{crs.attrs['grid_mapping_name']}_index"
        coords = call_on_dataset(
            lambda ds: (
                ds.drop_vars(list(grid_mapping_vars))
                .filter_by_attrs(standard_name=standard_name)
                .variables
            ),
            obj,
        )
        coord_names = list(coords)
        if not coord_names:
            raise ValueError(
                "Cannot find the cell index variable. Please specify it explicitly."
            )
        name = coord_names[0]

    translations = {"refinement_level": "level"}
    grid_info = {
        translations.get(name, name): value for name, value in crs.attrs.items()
    }
    grid_name = grid_info.pop("grid_mapping_name")
    var = vars_[name].copy(deep=False)
    var.attrs = grid_info

    if grid_name not in GRID_REGISTRY:
        raise ValueError(f"unknown grid name: {grid_name}")
    index_cls = GRID_REGISTRY[grid_name]

    index = index_cls.from_variables({name: var}, options=index_options)

    return xr.Coordinates({name: var}, indexes={name: index})
