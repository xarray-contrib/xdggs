import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self, TypeVar

import healpy
import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.grid import DGGSInfo
from xdggs.index import DGGSIndex
from xdggs.itertools import groupby, identity
from xdggs.utils import _extract_cell_id_variable, register_dggs

T = TypeVar("T")


@dataclass(frozen=True)
class HealpixInfo(DGGSInfo):
    resolution: int

    indexing_scheme: Literal["nested", "ring", "unique"] = "nested"

    rotation: tuple[float, float] = (0.0, 0.0)

    @property
    def nside(self: Self) -> int:
        return 2**self.resolution

    @property
    def nest(self: Self) -> bool:
        if self.indexing_scheme not in {"nested", "ring"}:
            raise ValueError(
                f"cannot convert indexing scheme {self.indexing_scheme} to `nest`"
            )
        else:
            return self.indexing_scheme == "nested"

    @classmethod
    def from_dict(cls: type[T], mapping: dict[str, Any]) -> T:
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

    def to_dict(self: Self) -> dict[str, Any]:
        return {
            "grid_name": "healpix",
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
        grid_info,
    ):
        super().__init__(cell_ids, dim)
        self._grid = grid_info

    @classmethod
    def from_variables(
        cls: type["HealpixIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "HealpixIndex":
        _, var, dim = _extract_cell_id_variable(variables)

        grid_info = HealpixInfo.from_dict(var.attrs | options)

        return cls(var.data, dim, grid_info)

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(new_pd_index, self._dim, self._grid)

    def _latlon2cellid(self, lat: Any, lon: Any) -> np.ndarray:
        # TODO apply rotation
        return healpy.ang2pix(
            self._grid.nside, lon, lat, lonlat=True, nest=self._grid.nest
        )

    def _cellid2latlon(self, cell_ids: Any) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = healpy.pix2ang(
            self._grid.nside, cell_ids, nest=self._grid.nest, lonlat=True
        )
        # TODO: apply rotation
        return np.stack([lon, lat], axis=0)

    def _repr_inline_(self, max_width: int):
        return f"HealpixIndex(nside={self._grid.resolution}, indexing_scheme={self._grid.indexing_scheme}, rotation={self._grid.rotation!r})"
