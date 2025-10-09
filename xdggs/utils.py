import xarray as xr

GRID_REGISTRY = {}


def register_dggs(name):
    def inner(cls):
        GRID_REGISTRY[name] = cls
        return cls

    return inner


def _extract_cell_id_variable(variables):
    # TODO: only one variable supported (raise otherwise)
    name, var = next(iter(variables.items()))

    # TODO: only 1-d variable supported (raise otherwise)
    dim = next(iter(var.dims))

    return name, var, dim


def call_on_dataset(func, obj, *args, kwargs=None):
    if kwargs is None:
        kwargs = {}

    if isinstance(obj, xr.DataArray):
        ds = obj._to_temp_dataset()
    else:
        ds = obj

    result = func(ds, *args, **kwargs)

    if isinstance(obj, xr.DataArray) and isinstance(result, xr.Dataset):
        return xr.DataArray._from_temp_dataset(result, name=obj.name)
    else:
        return result
