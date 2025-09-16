import xarray as xr


def call_on_dataset(func, obj, name, *args, kwargs=None):
    if kwargs is None:
        kwargs = {}

    if isinstance(obj, xr.DataArray):
        ds = obj._to_temp_dataset()
    else:
        ds = obj

    result = func(ds, *args, **kwargs)

    if isinstance(obj, xr.DataArray):
        return xr.DataArray._from_temp_dataset(result, name=obj.name)
    else:
        return result


def easygems(obj):
    orders = {"nested": "nest", "ring": "ring"}

    def _convert(ds):
        grid_info = ds.dggs.grid_info
        dim = ds.dggs.index._dim
        coord = ds.dggs._name

        order = orders.get(grid_info.indexing_scheme)
        if order is None:
            raise ValueError(f"easygems: unsupported indexing scheme: {order}")

        crs = xr.Variable(
            (),
            0,
            {
                "grid_mapping_name": "healpix",
                "healpix_nside": grid_info.nside,
                "healpix_order": order,
            },
        )

        return (
            ds.assign_coords(crs=crs)
            .drop_indexes(coord)
            .rename_dims({dim: "cell"})
            .rename_vars({coord: "cell"})
            .set_xindex("cell")
        )

    return call_on_dataset(_convert, obj)
