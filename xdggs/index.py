from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import xarray as xr
from xarray.indexes import Index, PandasIndex

from xdggs.grid import DGGSInfo
from xdggs.utils import GRID_REGISTRY, _extract_cell_id_variable

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping
    from typing import Any, Self

    from lonboard import BaseLayer as LonboardLayer
    from xarray.core.types import JoinOptions


class DGGSIndex(Index):
    _dim: str
    _index: xr.Index

    def __init__(self, cell_ids: Any | xr.Index, dim: str, grid_info: DGGSInfo):
        self._dim = dim

        if isinstance(cell_ids, xr.Index):
            self._index = cell_ids
        else:
            self._index = PandasIndex(cell_ids, dim)

        self._grid = grid_info

    @classmethod
    def from_variables(
        cls: type[DGGSIndex],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> DGGSIndex:
        name, var, _ = _extract_cell_id_variable(variables)

        grid_name = var.attrs["grid_name"]
        cls = GRID_REGISTRY.get(grid_name)
        if cls is None:
            raise ValueError(f"unknown DGGS grid name: {grid_name}")

        index = cls.from_variables(variables, options=options)
        if isinstance(index._index, PandasIndex):
            index._index.index.name = name

        return index

    @classmethod
    def full_domain(
        cls,
        level: int,
        dim: str,
        name: str,
        *,
        options: Mapping[str, Any],
    ) -> Self:
        """Create the index for the complete domain of the given level"""
        raise NotImplementedError("To be implemented in child class")

    def values(self):
        return self._index.index.values

    def equals(self, other: Index, **kwargs) -> bool:
        if (
            type(self) is not type(other)
            or self._dim != other._dim
            or self._grid != other._grid
        ):
            return False

        return self._index.equals(other._index, **kwargs)

    def create_variables(
        self, variables: Mapping[Any, xr.Variable] | None = None
    ) -> dict[Hashable, xr.Variable]:
        return self._index.create_variables(variables)

    def isel(
        self: DGGSIndex, indexers: Mapping[Any, int | np.ndarray | xr.Variable]
    ) -> DGGSIndex | None:
        new_index = self._index.isel(indexers)
        if new_index is not None:
            return self._replace(new_index)
        else:
            return None

    def sel(self, labels, method=None, **options):
        if method == "nearest":
            raise ValueError("finding nearest grid cell has no meaning")
        return self._index.sel(labels, method=method, **options)

    def join(self, other: Self, how: JoinOptions = "inner") -> Self:
        if self.grid_info != other.grid_info:
            raise ValueError(
                "Alignment with different grid parameters is not supported."
            )

        return self._replace(self._index.join(other._index, how=how))

    def reindex_like(self, other: Self) -> dict[Hashable, Any]:
        if self.grid_info != other.grid_info:
            raise ValueError(
                "Reindexing to different grid parameters is not supported."
            )

        return self._index.reindex_like(other._index)

    def _replace(self, new_index: PandasIndex):
        raise NotImplementedError()

    def serialize(self, *, encoding: dict[str, Any] | None = None) -> xr.Coordinates:
        unknown_encodings = [
            key for key in encoding if key not in {"compression", "coordinate"}
        ]
        if unknown_encodings:
            raise ValueError(
                f"Unknown encodings: {', '.join(repr(k) for k in unknown_encodings)}"
            )

        if encoding.get("compression", "none") != "none":
            raise ValueError(
                'This index implementation does not support any compression other than ``"none"``'
            )

        variables = self._index.create_variables()
        coords = xr.Coordinates(variables, indexes={})

        coordinate_name = encoding.get("coordinate", self.name)
        if coordinate_name != self.name:
            coords = coords.rename_vars({self.name: coordinate_name})
        return coords

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        return self._grid.cell_ids2geographic(self.values())

    def cell_boundaries(self) -> np.ndarray:
        return self.grid_info.cell_boundaries(self.values())

    def zoom_to(self, level: int) -> np.ndarray:
        return self._grid.zoom_to(self.values(), level=level)

    def _create_layer(
        self,
        cell_id_column: str,
        columns: dict[str, npt.NDArray],
        fill_colors: npt.NDArray[np.uint8],
    ) -> LonboardLayer:
        from arro3.core import Array
        from lonboard import SolidPolygonLayer

        from xdggs.plotting.arrow import create_arrow_table

        polygons = self.grid_info.cell_boundaries(
            columns[cell_id_column], backend="geoarrow"
        )
        table = create_arrow_table(columns | {"geometry": Array.from_arrow(polygons)})

        return SolidPolygonLayer(table=table, filled=True, get_fill_color=fill_colors)

    @property
    def grid_info(self) -> DGGSInfo:
        return self._grid

    @property
    def dim(self) -> str:
        return self._dim

    @property
    def name(self) -> str:
        return self._index.index.name
