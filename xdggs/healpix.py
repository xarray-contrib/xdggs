import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import healpy
import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.index import DGGSIndex
from xdggs.itertools import groupby, identity
from xdggs.utils import _extract_cell_id_variable, register_dggs


@dataclass(frozen=True)
class HealpixInfo:
    resolution: int

    indexing_scheme: Literal["nested", "ring", "unique"] = "nested"

    rotation: tuple[float, float] = (0.0, 0.0)

    @property
    def nside(self):
        return 2**self.resolution

    @property
    def nest(self):
        if self.indexing_scheme not in {"nested", "ring"}:
            raise ValueError(
                f"cannot convert index scheme {self.indexing_scheme} to `nest`"
            )
        else:
            return self.indexing_scheme == "nested"

    @classmethod
    def from_dict(cls, mapping):
        translations = {
            "nside": ("resolution", lambda nside: int(np.log2(nside))),
            "order": ("resolution", identity),
            "level": ("resolution", identity),
            "depth": ("resolution", identity),
            "nest": ("indexing_scheme", lambda nest: "nested" if nest else "ring"),
            "rot_latlon": (
                "rotation",
                lambda rot_latlon: (rot_latlon[1], rot_latlon[0]),
            ),
        }

        def translate(name, value):
            new_name, translator = translations.get(name, (name, identity))

            return new_name, name, translator(value)

        translated = (translate(name, value) for name, value in mapping.items())

        grouped = {
            name: [(old_name, value) for _, old_name, value in group]
            for name, group in groupby(translated, key=operator.itemgetter(0))
        }
        duplicated_parameters = {
            name: group for name, group in grouped.items() if len(group) != 1
        }
        if duplicated_parameters:
            raise ExceptionGroup(
                "received multiple values for parameters",
                [
                    ValueError(
                        f"Parameter {name} received multiple values: {[n for n, _ in group]}"
                    )
                    for name, group in duplicated_parameters.items()
                ],
            )

        params = {
            name: group[0][1] for name, group in grouped.items() if name != "grid_name"
        }

        return cls(**params)

    def to_dict(self):
        return {
            "grid_type": "healpix",
            "resolution": self.resolution,
            "indexing_scheme": self.indexing_scheme,
            "rotation": list(self.rotation),
        }


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
        return healpy.ang2pix(self._nside, lon, lat, lonlat=True, nest=self._nest)

    def _cellid2latlon(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = healpy.pix2ang(self._nside, cell_ids, nest=self._nest, lonlat=True)
        return np.stack([lon, lat], axis=-1)

    def _repr_inline_(self, max_width: int):
        return f"HealpixIndex(nside={self._nside}, nest={self._nest}, rot_latlon={self._rot_latlon!r})"
