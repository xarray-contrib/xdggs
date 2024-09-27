from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeVar

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import healpy
import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.grid import DGGSInfo, translate_parameters
from xdggs.index import DGGSIndex
from xdggs.itertools import identity
from xdggs.utils import _extract_cell_id_variable, register_dggs

T = TypeVar("T")


@dataclass(frozen=True)
class HealpixInfo(DGGSInfo):
    level: int

    indexing_scheme: Literal["nested", "ring", "unique"] = "nested"

    rotation: list[float, float] = field(default_factory=lambda: [0.0, 0.0])

    valid_parameters: ClassVar[dict[str, Any]] = {
        "level": range(0, 29 + 1),
        "indexing_scheme": ["nested", "ring", "unique"],
    }

    def __post_init__(self):
        if self.level not in self.valid_parameters["level"]:
            raise ValueError("level must be an integer in the range of [0, 29]")

        if self.indexing_scheme not in self.valid_parameters["indexing_scheme"]:
            raise ValueError(
                f"indexing scheme must be one of {self.valid_parameters['indexing_scheme']}"
            )

        if np.any(np.isnan(self.rotation) | np.isinf(self.rotation)):
            raise ValueError(
                f"rotation must consist of finite values, got {self.rotation}"
            )

    @property
    def nside(self: Self) -> int:
        return 2**self.level

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
        def translate_nside(nside):
            log = np.log2(nside)
            potential_level = int(log)
            if potential_level != log:
                raise ValueError("`nside` has to be an integer power of 2")

            return potential_level

        translations = {
            "nside": ("level", translate_nside),
            "order": ("level", identity),
            "resolution": ("level", identity),
            "depth": ("level", identity),
            "nest": ("indexing_scheme", lambda nest: "nested" if nest else "ring"),
            "rot_latlon": (
                "rotation",
                lambda rot_latlon: (rot_latlon[1], rot_latlon[0]),
            ),
        }

        params = translate_parameters(mapping, translations)
        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        return {
            "grid_name": "healpix",
            "level": self.level,
            "indexing_scheme": self.indexing_scheme,
            "rotation": self.rotation,
        }

    def cell_ids2geographic(self, cell_ids):
        lon, lat = healpy.pix2ang(self.nside, cell_ids, nest=self.nest, lonlat=True)

        return lon, lat

    def geographic2cell_ids(self, lon, lat):
        return healpy.ang2pix(self.nside, lon, lat, lonlat=True, nest=self.nest)


@register_dggs("healpix")
class HealpixIndex(DGGSIndex):
    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        grid_info: DGGSInfo,
    ):
        if not isinstance(grid_info, HealpixInfo):
            raise ValueError(f"grid info object has an invalid type: {type(grid_info)}")

        super().__init__(cell_ids, dim, grid_info)

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

    @property
    def grid_info(self) -> HealpixInfo:
        return self._grid

    def _repr_inline_(self, max_width: int):
        return f"HealpixIndex(nside={self._grid.level}, indexing_scheme={self._grid.indexing_scheme}, rotation={self._grid.rotation!r})"
