from collections.abc import Hashable
from typing import Any, Literal

import numpy as np
import xarray as xr

from xdggs.conventions.base import Convention
from xdggs.conventions.errors import DecoderError
from xdggs.conventions.registry import register_convention
from xdggs.conventions.utils import infer_grid_name
from xdggs.utils import GRID_REGISTRY


def remove_grid_mapping(ds, name):
    new = ds.copy(deep=False)
    for var in new.variables.values():
        if var.attrs.get("grid_mapping") != name:
            continue

        del var.attrs["grid_mapping"]

    return new


@register_convention("cf")
class Cf(Convention):
    def translate_keys(
        self,
        mapping: dict[str, Any],
        direction: Literal["forward", "inverse"] = "forward",
    ) -> dict[str, Any]:
        translations = {"grid_mapping_name": "grid_name", "refinement_level": "level"}
        if direction == "inverse":
            translations = {v: k for k, v in translations.items()}

        return {translations.get(key, key): value for key, value in mapping.items()}

    def decode(
        self,
        ds: xr.Dataset,
        *,
        grid_info: dict[str, Any] | None = None,
        name: Hashable | None = None,
        index_options: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        grid_mapping_vars = {
            name: var
            for name, var in ds.variables.items()
            if "grid_mapping_name" in var.attrs
        }
        if len(grid_mapping_vars) != 1:
            raise DecoderError(
                "cf convention: requires exactly one grid mapping variable for now."
                f" Got {len(grid_mapping_vars)}"
            )
        crs_name, crs = next(iter(grid_mapping_vars.items()))

        if name is None:
            standard_name = f"{crs.attrs['grid_mapping_name']}_index"
            coords = (
                ds.drop_vars(list(grid_mapping_vars))
                .filter_by_attrs(standard_name=standard_name)
                .variables
            )
            coord_names = list(coords)
            if not coord_names:
                raise DecoderError(
                    "cf convention: Cannot find the cell index variable."
                    " Please specify it explicitly."
                )
            name = coord_names[0]

        grid_info = self.translate_keys(crs.attrs, direction="forward")
        grid_name = grid_info["grid_name"]

        var = ds.variables[name].copy(deep=False)
        var.attrs = grid_info

        if grid_name not in GRID_REGISTRY:
            raise DecoderError(f"cf convention: unknown grid name: {grid_name}")
        index_cls = GRID_REGISTRY[grid_name]

        index = index_cls.from_variables({name: var}, options=index_options)
        return (
            ds.drop_vars([crs_name, name])
            .pipe(remove_grid_mapping, grid_name)
            .assign_coords(xr.Coordinates.from_xindex(index))
        )

    def encode(self, ds: xr.Dataset, *, encoding: dict[str, Any] | None = None):
        if encoding is None:
            encoding = {}

        crs_name = encoding.get("grid_mapping_variable", "crs")

        grid_info = ds.dggs.grid_info
        dim = ds.dggs.index._dim
        name = ds.dggs._name
        coord = ds.dggs.coord.variable

        grid_name = infer_grid_name(ds.dggs.index)
        grid_info_dict = grid_info.to_dict()
        metadata = self.translate_keys(grid_info_dict, direction="inverse")

        crs = xr.Variable((), np.int8(0), metadata)

        additional_var_attrs = {"coordinates": coord, "grid_mapping": grid_name}
        coord_attrs = {"standard_name": f"{grid_name}_index", "units": "1"}

        new = ds.drop_indexes(name).drop_vars(name).copy(deep=False)
        for var in new.variables.values():
            if dim not in var.dims:
                continue

            var.attrs |= additional_var_attrs

        existing_attrs = {
            name: value
            for name, value in coord.attrs.items()
            if name not in grid_info_dict
        }
        coord.attrs = existing_attrs | coord_attrs

        coords = xr.Coordinates({crs_name: crs, name: coord}, indexes={})
        return new.assign_coords(coords)
