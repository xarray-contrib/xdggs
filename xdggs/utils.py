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
