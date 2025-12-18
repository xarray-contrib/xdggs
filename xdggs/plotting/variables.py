import xarray as xr


def construct_variable_chooser(obj):
    from ipywidgets import Dropdown

    if isinstance(obj, xr.DataArray):
        options = [obj.name] if obj.name is not None else []
        value = obj.name
    else:
        options = [
            name
            for name, var in obj.data_vars.items()
            if obj.dggs.index._dim in var.dims
        ]
        value = options[0] if options else None

    return Dropdown(
        options=options,
        value=value,
        description="Variable:",
        disabled=len(options) <= 1,
    )
