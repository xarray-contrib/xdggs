import numpy as np
import xarray as xr

from xdggs.conventions.registry import Convention, register_convention
from xdggs.conventions.utils import infer_grid_name
from xdggs.utils import GRID_REGISTRY, call_on_dataset


@register_convention("cf")
class Cf(Convention):
    def decode(self, obj, grid_info, name, index_options):
        vars_ = call_on_dataset(
            lambda ds: ds.variables,
            obj,
        )
        grid_mapping_vars = {
            name: var for name, var in vars_.items() if "grid_mapping_name" in var.attrs
        }
        if len(grid_mapping_vars) != 1:
            raise ValueError(
                "needs exactly one grid mapping variable for now."
                f" Got {len(grid_mapping_vars)}"
            )
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

        translations = {"grid_mapping_name": "grid_name", "refinement_level": "level"}
        grid_info = {
            translations.get(name, name): value for name, value in crs.attrs.items()
        }
        grid_name = grid_info.pop("grid_name")
        var = vars_[name].copy(deep=False)
        var.attrs = grid_info

        if grid_name not in GRID_REGISTRY:
            raise ValueError(f"unknown grid name: {grid_name}")
        index_cls = GRID_REGISTRY[grid_name]

        index = index_cls.from_variables({name: var}, options=index_options)
        return xr.Coordinates.from_xindex(index)

    def encode(self, obj):
        def _convert(ds):
            grid_info = ds.dggs.grid_info
            dim = ds.dggs.index._dim
            coord = ds.dggs._name

            grid_name = infer_grid_name(ds.dggs.index)
            metadata = grid_info.to_dict() | {"grid_mapping_name": grid_name}
            metadata["refinement_level"] = metadata.pop("level")
            metadata.pop("grid_name", None)

            crs = xr.Variable((), np.int8(0), metadata)

            additional_var_attrs = {"coordinates": coord, "grid_mapping": "crs"}
            coord_attrs = {"standard_name": f"{grid_name}_index", "units": "1"}

            new = ds.drop_indexes(coord).copy(deep=False)
            for key, var in new.variables.items():
                if key == coord or dim not in var.dims:
                    continue

                var.attrs |= additional_var_attrs

            grid_info_dict = grid_info.to_dict()
            new_attrs = {
                name: value
                for name, value in new[coord].attrs.items()
                if name not in grid_info_dict
            }
            new[coord].attrs = new_attrs | coord_attrs

            return new.assign_coords({"crs": crs})

        return call_on_dataset(_convert, obj)
