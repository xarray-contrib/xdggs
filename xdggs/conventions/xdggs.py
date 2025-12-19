from collections.abc import Hashable
from typing import Any

import xarray as xr

from xdggs.conventions.base import Convention
from xdggs.conventions.errors import DecoderError
from xdggs.conventions.registry import register_convention
from xdggs.conventions.utils import infer_grid_name
from xdggs.grid import DGGSInfo
from xdggs.utils import GRID_REGISTRY


@register_convention("xdggs")
class Xdggs(Convention):
    def decode(
        self,
        ds: xr.Dataset,
        *,
        grid_info: dict[str, Any] | DGGSInfo | None = None,
        name: Hashable | None = None,
        index_options: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        if name is None:
            name = "cell_ids"

        try:
            var = ds[name]
        except KeyError:
            raise DecoderError(
                f"xdggs convention: Cannot find the cell ids coordinate ({name})"
            )

        if len(var.dims) != 1:
            # TODO: allow 0D
            raise DecoderError(
                "xdggs convention: cell id coordinate must be 1D"
                f" but has dims {tuple(var.dims)}"
            )
        [dim] = var.dims

        if grid_info is None:
            grid_info = var.attrs
        elif isinstance(grid_info, DGGSInfo):
            # TODO: avoid serializing / deserializing cycle
            grid_info = grid_info.to_dict()

        grid_name = grid_info["grid_name"]
        if grid_name not in GRID_REGISTRY:
            raise DecoderError(f"xdggs convention: unknown grid name: {grid_name}")
        index_cls = GRID_REGISTRY[grid_name]

        var_ = var.copy(deep=True)
        var_.attrs = grid_info
        index = index_cls.from_variables({name: var_}, options=index_options)

        return ds.assign_coords(xr.Coordinates.from_xindex(index))

    def encode(self, ds: xr.Dataset, *, encoding: None = None) -> xr.Dataset:
        coord = ds.dggs._name

        grid_name = infer_grid_name(ds.dggs.index)
        metadata = {"grid_name": grid_name} | ds.dggs.grid_info.to_dict()

        # TODO: `assign_coords` + `assign_attrs` drops the index
        ds_ = ds.copy(deep=False)
        ds_[coord].attrs.update(metadata)
        return ds_
