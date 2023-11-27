from collections.abc import Mapping
from typing import Any

import healpy
import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs


@register_dggs("healpix")
class HealpixIndex(DGGSIndex):
    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        nside: int,
        nest: bool,
        rot_latlon: tuple[float, float],
    ):
        super().__init__(cell_ids, dim)

        self._nside = nside
        self._nest = nest
        self._rot_latlon = rot_latlon

    @classmethod
    def from_variables(
        cls: type["HealpixIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "HealpixIndex":
        _, var, dim = _extract_cell_id_variable(variables)

        nside = var.attrs.get("nside", options.get("nside"))
        nest = var.attrs.get("nest", options.get("nest", False))
        rot_latlon = var.attrs.get("rot_latlon", options.get("rot_latlon", (0.0, 0.0)))

        return cls(var.data, dim, nside, nest, rot_latlon)

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(
            new_pd_index, self._dim, self._nside, self._nest, self._rot_latlon
        )

    def _latlon2cellid(self, lat: Any, lon: Any) -> np.ndarray:
        return healpy.ang2pix(self._nside, -lon, lat, lonlat=True, nest=self._nest)

    def _cellid2latlon(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = healpy.pix2ang(self._nside, cell_ids, nest=self._nest, lonlat=True)
        return lat, -lon

    def _repr_inline_(self, max_width: int):
        return f"HealpixIndex(nside={self._nside}, nest={self._nest}, rot_latlon={self._rot_latlon!r})"
