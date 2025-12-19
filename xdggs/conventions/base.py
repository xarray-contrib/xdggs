from collections.abc import Hashable
from typing import Any

import xarray as xr

from xdggs.grid import DGGSInfo


class Convention:
    def decode(
        self,
        obj: xr.Dataset | xr.DataArray,
        *,
        grid_info: DGGSInfo | None,
        name: Hashable | None,
        index_options: dict[str, Any] | None,
    ):
        raise NotImplementedError

    def encode(
        self, obj: xr.Dataset | xr.DataArray, *, encoding: dict[str, Any] | None = None
    ):
        raise NotImplementedError
