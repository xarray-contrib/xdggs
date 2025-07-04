import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import numpy as np
import xarray as xr

try:
    from h3ronpy.vector import (
        cells_to_coordinates,
        cells_to_wkb_polygons,
        coordinates_to_cells,
    )
except ImportError:
    from h3ronpy.arrow.vector import (
        cells_to_coordinates,
        cells_to_wkb_polygons,
        coordinates_to_cells,
    )
from xarray.indexes import PandasIndex

from xdggs.grid import DGGSInfo, translate_parameters
from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs


def polygons_shapely(wkb):
    import shapely

    return shapely.from_wkb(wkb)


def polygons_geoarrow(wkb):
    import pyproj
    import shapely
    from arro3.core import list_array

    polygons = shapely.from_wkb(wkb)
    crs = pyproj.CRS.from_epsg(4326)

    geometry_type, coords, (ring_offsets, geom_offsets) = shapely.to_ragged_array(
        polygons
    )

    if geometry_type != shapely.GeometryType.POLYGON:
        raise ValueError(f"unsupported geometry type found: {geometry_type}")

    polygon_array = list_array(
        geom_offsets.astype("int32"), list_array(ring_offsets.astype("int32"), coords)
    )
    polygon_array_with_geo_meta = polygon_array.cast(
        polygon_array.field.with_metadata(
            {
                "ARROW:extension:name": "geoarrow.polygon",
                "ARROW:extension:metadata": json.dumps({"crs": crs.to_json_dict()}),
            }
        )
    )

    return polygon_array_with_geo_meta


@dataclass(frozen=True)
class H3Info(DGGSInfo):
    """
    Grid information container for h3 grids.

    Parameters
    ----------
    level : int
        Grid hierarchical level. A higher value corresponds to a finer grid resolution
        with smaller cell areas. The number of cells covering the whole sphere usually
        grows exponentially with increasing level values, ranging from 5-100 cells at
        level 0 to millions or billions of cells at level 10+ (the exact numbers depends
        on the specific grid).
    """

    level: int
    """int : The hierarchical level of the grid"""

    valid_parameters: ClassVar[dict[str, Any]] = {"level": range(16)}

    def __post_init__(self):
        if self.level not in self.valid_parameters["level"]:
            raise ValueError("level must be an integer between 0 and 15")

    @classmethod
    def from_dict(cls: type[Self], mapping: dict[str, Any]) -> Self:
        """construct a `H3Info` object from a mapping of attributes

        Parameters
        ----------
        mapping: mapping of str to any
            The attributes.

        Returns
        -------
        grid_info : H3Info
            The constructed grid info object.
        """
        translations = {
            "resolution": ("level", int),
            "level": ("level", int),
        }

        params = translate_parameters(mapping, translations)
        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        """
        Dump the normalized grid parameters.

        Returns
        -------
        mapping : dict of str to any
            The normalized grid parameters.
        """
        return {"grid_name": "h3", "level": self.level}

    def cell_ids2geographic(
        self, cell_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert cell ids to geographic coordinates

        Parameters
        ----------
        cell_ids : array-like
            Array-like containing the cell ids.

        Returns
        -------
        lon : array-like
            The longitude coordinate values of the grid cells in degree
        lat : array-like
            The latitude coordinate values of the grid cells in degree
        """
        result = cells_to_coordinates(cell_ids, radians=False)

        lon = result.column("lng").to_numpy()
        lat = result.column("lat").to_numpy()

        return lon, lat

    def geographic2cell_ids(self, lon, lat):
        """
        Convert cell ids to geographic coordinates

        This will perform a binning operation: any point within a grid cell will be assign
        that cell's ID.

        Parameters
        ----------
        lon : array-like
            The longitude coordinate values in degree
        lat : array-like
            The latitude coordinate values in degree

        Returns
        -------
        cell_ids : array-like
            Array-like containing the cell ids.
        """
        return coordinates_to_cells(lat, lon, self.level, radians=False)

    def cell_boundaries(self, cell_ids, backend="shapely"):
        """
        Derive cell boundary polygons from cell ids

        Parameters
        ----------
        cell_ids : array-like
            The cell ids.
        backend : {"shapely", "geoarrow"}, default: "shapely"
            The backend to convert to.

        Returns
        -------
        polygons : array-like
            The derived cell boundary polygons. The format differs based on the passed
            backend:

            - ``"shapely"``: return a array of :py:class:`shapely.Polygon` objects
            - ``"geoarrow"``: return a ``geoarrow`` array
        """
        # TODO: convert cell ids directly to geoarrow once h3ronpy supports it
        wkb = cells_to_wkb_polygons(cell_ids, radians=False, link_cells=False)

        backends = {
            "shapely": polygons_shapely,
            "geoarrow": polygons_geoarrow,
        }
        backend_func = backends.get(backend)
        if backend_func is None:
            raise ValueError(f"invalid backend: {backend!r}")
        return backend_func(wkb)


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
