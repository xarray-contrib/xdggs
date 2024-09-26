from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import operator

import numpy as np
import xarray as xr
from h3ronpy.arrow.vector import cells_to_coordinates, coordinates_to_cells
from xarray.indexes import PandasIndex

from xdggs.grid import DGGSInfo
from xdggs.index import DGGSIndex
from xdggs.itertools import groupby, identity
from xdggs.utils import _extract_cell_id_variable, register_dggs


@dataclass(frozen=True)
class H3Info(DGGSInfo):
    level: int

    valid_parameters: ClassVar[dict[str, Any]] = {"level": range(16)}

    def __post_init__(self):
        if self.level not in self.valid_parameters["level"]:
            raise ValueError("level must be an integer between 0 and 15")

    @classmethod
    def from_dict(cls: type[Self], mapping: dict[str, Any]) -> Self:
        translations = {
            "resolution": ("level", identity),
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
                        f"Parameter {name} received multiple values: {sorted(n for n, _ in group)}"
                    )
                    for name, group in duplicated_parameters.items()
                ],
            )

        params = {
            name: group[0][1] for name, group in grouped.items() if name != "grid_name"
        }
        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        return {"grid_name": "h3", "level": self.level}

    def cell_ids2geographic(
        self, cell_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lat, lon = cells_to_coordinates(cell_ids, radians=False)

        return lon, lat

    def geographic2cell_ids(self, lon, lat):
        return coordinates_to_cells(lat, lon, self.level, radians=False)


@register_dggs("h3")
class H3Index(DGGSIndex):
    _grid: DGGSInfo

    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        grid_info: DGGSInfo,
    ):
        super().__init__(cell_ids, dim, grid_info)

    @classmethod
    def from_variables(
        cls: type["H3Index"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "H3Index":
        _, var, dim = _extract_cell_id_variable(variables)

        grid_info = H3Info.from_dict(var.attrs | options)

        return cls(var.data, dim, grid_info)

    @property
    def grid_info(self) -> H3Info:
        return self._grid

    def _replace(self, new_pd_index: PandasIndex):
        return type(self)(new_pd_index, self._dim, self._grid)

    def _repr_inline_(self, max_width: int):
        return f"H3Index(level={self._grid.level})"
