from xarray.indexes import Index, PandasIndex

from .utils import GRID_REGISTRY, _extract_cell_id_variable


def _extract_cell_id_variable(variables):
    # TODO: only one variable supported (raise otherwise)
    name, var = next(iter(variables.items()))

    # TODO: only 1-d variable supported (raise otherwise)
    dim = next(iter(var.dims))

    return name, var, dim


class DGGSIndex(Index):

    def __init__(self, cell_ids, dim):
        self._dim = dim

        if isinstance(cell_ids, PandasIndex):
            self._pd_index = cell_ids
        else:
            self._pd_index = PandasIndex(cell_ids, dim)

    @classmethod
    def from_variables(cls, variables, *, options):
        name, var, dim = _extract_cell_id_variable(variables)

        grid_name = var.attrs["grid_name"]
        cls = GRID_REGISTRY[grid_name]

        return cls.from_variables(variables, options=options)
    
    def create_variables(self, variables=None):
        return self._pd_index.create_variables(variables)

    def isel(self, indexers):
        new_pd_index = self._pd_index.isel(indexers)
        if new_pd_index is not None:
            return self._replace(new_pd_index)
        else:
            return None
    
    def sel(self, labels, method=None, tolerance=None):
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        return self._pd_index.sel(labels, method=method, tolerance=tolerance)

    def _replace(self, new_pd_index):
        raise NotImplementedError()
    
    def _latlon2cellid(self, lat, lon):
        """convert latitude / longitude points to cell ids."""
        raise NotImplementedError()

    def _cellid2latlon(self, cell_ids):
        """convert cell centers to latitude / longitude."""
        raise NotImplementedError()

    @property
    def cell_centers(self):
        return self._cellid2latlon(self._pd_index.index.values)