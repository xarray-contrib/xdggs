import json
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self

import numpy as np

from xdggs.ellipsoid import (
    Ellipsoid,
    EllipsoidLike,
    Sphere,
    SphereLike,
    parse_ellipsoid,
)
from xdggs.grid import DGGSInfo, translate_parameters
from xdggs.itertools import identity
from xdggs.utils import ignore_parameters


def _serialize_ellipsoid(
    ellipsoid: SphereLike | EllipsoidLike | None,
) -> SphereLike | EllipsoidLike:
    if ellipsoid is None:
        import healpix_geo.ellipsoid

        return healpix_geo.ellipsoid.resolve("sphere")
    elif not isinstance(ellipsoid, dict):
        return ellipsoid.to_dict()
    else:
        return ellipsoid


def polygons_shapely(vertices):
    import shapely

    return shapely.polygons(vertices)


def polygons_geoarrow(vertices):
    import pyproj
    from arro3.core import list_array

    polygon_vertices = np.concatenate([vertices, vertices[:, :1, :]], axis=1)
    crs = pyproj.CRS.from_epsg(4326)

    # construct geoarrow arrays
    coords = np.reshape(polygon_vertices, (-1, 2))
    coords_per_pixel = polygon_vertices.shape[1]
    geom_offsets = np.arange(vertices.shape[0] + 1, dtype="int32")
    ring_offsets = geom_offsets * coords_per_pixel

    polygon_array = list_array(geom_offsets, list_array(ring_offsets, coords))

    # We need to tag the array with extension metadata (`geoarrow.polygon`) so that Lonboard knows that this is a geospatial column.
    polygon_array_with_geo_meta = polygon_array.cast(
        polygon_array.field.with_metadata(
            {
                "ARROW:extension:name": "geoarrow.polygon",
                "ARROW:extension:metadata": json.dumps(
                    {"crs": crs.to_json_dict(), "edges": "spherical"}
                ),
            }
        )
    )
    return polygon_array_with_geo_meta


def center_around_prime_meridian(lon, lat):
    # three tasks:
    # - center around the prime meridian (map to a range of [-180, 180])
    # - replace the longitude of points at the poles with the median
    #   of longitude of the other vertices
    # - cells that cross the dateline should have longitudes around 180

    # center around prime meridian
    recentered = (lon + 180) % 360 - 180

    # replace lon of pole with the median of the remaining vertices
    contains_poles = np.isin(lat, np.array([-90, 90]))
    pole_cells = np.any(contains_poles, axis=-1)
    recentered[contains_poles] = np.median(
        np.reshape(
            recentered[pole_cells[:, None] & np.logical_not(contains_poles)], (-1, 3)
        ),
        axis=-1,
    )

    # keep cells that cross the dateline centered around 180
    polygons_to_fix = np.any(recentered < -100, axis=-1) & np.any(
        recentered > 100, axis=-1
    )
    result = np.where(
        polygons_to_fix[:, None] & (recentered < 0), recentered + 360, recentered
    )

    return result


@dataclass(frozen=True)
class HealpixInfo(DGGSInfo):
    """
    Grid information container for healpix grids.

    Parameters
    ----------
    level : int
        Grid hierarchical level. A higher value corresponds to a finer grid resolution
        with smaller cell areas. The number of cells covering the whole sphere usually
        grows exponentially with increasing level values, ranging from 5-100 cells at
        level 0 to millions or billions of cells at level 10+ (the exact numbers depends
        on the specific grid).
    indexing_scheme : {"nested", "ring", "zuniq", "nuniq"}, default: "nested"
        The indexing scheme of the healpix grid.

        .. warning::
            Note that ``"nuniq"`` is currently not supported as the underlying library
            (:doc:`healpix-geo <healpix-geo:index>`) does not support it.
    ellipsoid : ellipsoid-like, optional
        The reference ellipsoid. If not passed, a sphere is assumed.
    """

    level: int | None
    """int or None : The hierarchical level of the grid"""

    indexing_scheme: Literal["nested", "ring", "zuniq"] = "nested"
    """int : The indexing scheme of the grid"""

    ellipsoid: str | SphereLike | EllipsoidLike | None = None
    """The ellipsoid"""

    valid_parameters: ClassVar[dict[str, Any]] = {
        "level": range(0, 29 + 1),
        "indexing_scheme": ["nested", "ring", "zuniq", "nuniq"],
    }

    def __post_init__(self):
        import healpix_geo.ellipsoid

        if self.indexing_scheme not in self.valid_parameters["indexing_scheme"]:
            raise ValueError(
                f"indexing scheme must be one of {self.valid_parameters['indexing_scheme']}"
            )
        elif self.indexing_scheme == "nuniq":
            raise ValueError("the indexing scheme `nuniq` is currently not supported")

        if self.indexing_scheme in {"zuniq", "nuniq"}:
            if self.level is not None:
                raise ValueError("level must be `None` for uniq indexing schemes")
        elif self.level not in self.valid_parameters["level"]:
            raise ValueError("level must be an integer in the range of [0, 29]")

        if isinstance(self.ellipsoid, str):
            object.__setattr__(
                self, "ellipsoid", healpix_geo.ellipsoid.resolve(self.ellipsoid)
            )

    @property
    def nside(self: Self) -> int:
        """resolution as the healpy-compatible nside parameter"""
        return 2**self.level

    @property
    def nest(self: Self) -> bool:
        """indexing_scheme as the healpy-compatible nest parameter"""
        if self.indexing_scheme not in {"nested", "ring"}:
            raise ValueError(
                f"cannot convert indexing scheme {self.indexing_scheme} to `nest`"
            )
        else:
            return self.indexing_scheme == "nested"

    @classmethod
    def from_dict[T](cls: type[T], mapping: dict[str, Any]) -> T:
        """construct a `HealpixInfo` object from a mapping of attributes

        Parameters
        ----------
        mapping: mapping of str to any
            The attributes.

        Returns
        -------
        grid_info : HealpixInfo
            The constructed grid info object.
        """

        def translate_nside(nside):
            log = np.log2(nside)
            potential_level = int(log)
            if potential_level != log:
                raise ValueError("`nside` has to be an integer power of 2")

            return potential_level

        def translate_ellipsoid(value):
            if isinstance(value, (str, Sphere, Ellipsoid)):
                return value
            elif value is None or not value:
                return value

            return parse_ellipsoid(value)

        translations = {
            "nside": ("level", translate_nside),
            "order": ("level", identity),
            "resolution": ("level", identity),
            "depth": ("level", identity),
            "nest": ("indexing_scheme", lambda nest: "nested" if nest else "ring"),
            "ellipsoid": ("ellipsoid", translate_ellipsoid),
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
        optional_values = {}
        if self.ellipsoid is not None:
            optional_values["ellipsoid"] = _serialize_ellipsoid(self.ellipsoid)

        return {
            "grid_name": "healpix",
            "level": self.level,
            "indexing_scheme": self.indexing_scheme,
        } | optional_values

    def cell_ids2geographic(self, cell_ids):
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
        import healpix_geo

        converters = {
            "nested": healpix_geo.nested.healpix_to_lonlat,
            "ring": healpix_geo.ring.healpix_to_lonlat,
            "zuniq": ignore_parameters("depth")(healpix_geo.zuniq.healpix_to_lonlat),
        }
        converter = converters[self.indexing_scheme]

        return converter(
            cell_ids, depth=self.level, ellipsoid=_serialize_ellipsoid(self.ellipsoid)
        )

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
        import healpix_geo

        if self.indexing_scheme in {"zuniq", "nuniq"}:
            raise ValueError(
                "Converting geographic coordinates to `uniq` schemes is not supported."
                " Please convert to a `nested` scheme and convert that to"
                " the desired uniq scheme."
            )

        converters = {
            "nested": healpix_geo.nested.lonlat_to_healpix,
            "ring": healpix_geo.ring.lonlat_to_healpix,
        }
        converter = converters[self.indexing_scheme]

        return converter(
            lon, lat, depth=self.level, ellipsoid=_serialize_ellipsoid(self.ellipsoid)
        )

    def cell_boundaries(self, cell_ids: Any, backend="shapely") -> np.ndarray:
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
        import healpix_geo

        converters = {
            "nested": healpix_geo.nested.vertices,
            "ring": healpix_geo.ring.vertices,
            "zuniq": ignore_parameters("depth")(healpix_geo.zuniq.vertices),
        }
        converter = converters[self.indexing_scheme]

        lon, lat = converter(
            cell_ids, depth=self.level, ellipsoid=_serialize_ellipsoid(self.ellipsoid)
        )

        lon_reshaped = np.reshape(lon, (-1, 4))
        lat_reshaped = np.reshape(lat, (-1, 4))

        lon_ = center_around_prime_meridian(lon_reshaped, lat_reshaped)

        vertices = np.stack((lon_, lat_reshaped), axis=-1)

        backends = {
            "shapely": polygons_shapely,
            "geoarrow": polygons_geoarrow,
        }

        backend_func = backends.get(backend)
        if backend_func is None:
            raise ValueError(f"invalid backend: {backend!r}")

        return backend_func(vertices)

    def zoom_to(self, cell_ids, level):
        if self.indexing_scheme == "ring":
            raise ValueError(
                "Scaling does not make sense for the 'ring' scheme."
                " Please convert to a nested scheme first."
            )
        elif self.indexing_scheme == "zuniq":
            raise NotImplementedError(
                "Zooming cell ids in the 'zuniq' scheme currently not supported"
            )

        from healpix_geo.nested import zoom_to

        return zoom_to(cell_ids, self.level, level)
