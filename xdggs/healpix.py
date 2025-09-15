import json
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self, TypeVar

import cdshealpix.nested
import cdshealpix.ring
import numpy as np
import xarray as xr
from healpix_geo.nested import RangeMOCIndex
from xarray.core.indexes import IndexSelResult, PandasIndex

from xdggs.grid import DGGSInfo, translate_parameters
from xdggs.index import DGGSIndex
from xdggs.itertools import identity
from xdggs.utils import _extract_cell_id_variable, register_dggs

T = TypeVar("T")

try:
    import dask.array as da

    dask_array_type = (da.Array,)
except ImportError:
    dask_array_type = ()


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
    indexing_scheme : {"nested", "ring", "unique"}, default: "nested"
        The indexing scheme of the healpix grid.

        .. warning::
            Note that ``"unique"`` is currently not supported as the underlying library
            (:doc:`cdshealpix <cdshealpix-python:index>`) does not support it.
    """

    level: int
    """int : The hierarchical level of the grid"""

    indexing_scheme: Literal["nested", "ring"] = "nested"
    """int : The indexing scheme of the grid"""

    valid_parameters: ClassVar[dict[str, Any]] = {
        "level": range(0, 29 + 1),
        "indexing_scheme": ["nested", "ring"],
    }

    def __post_init__(self):
        if self.level not in self.valid_parameters["level"]:
            raise ValueError("level must be an integer in the range of [0, 29]")

        if self.indexing_scheme not in self.valid_parameters["indexing_scheme"]:
            raise ValueError(
                f"indexing scheme must be one of {self.valid_parameters['indexing_scheme']}"
            )
        elif self.indexing_scheme == "unique":
            raise ValueError("the indexing scheme `unique` is currently not supported")

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
    def from_dict(cls: type[T], mapping: dict[str, Any]) -> T:
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

        translations = {
            "nside": ("level", translate_nside),
            "order": ("level", identity),
            "resolution": ("level", identity),
            "depth": ("level", identity),
            "nest": ("indexing_scheme", lambda nest: "nested" if nest else "ring"),
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
        return {
            "grid_name": "healpix",
            "level": self.level,
            "indexing_scheme": self.indexing_scheme,
        }

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
        converters = {
            "nested": cdshealpix.nested.healpix_to_lonlat,
            "ring": lambda cell_ids, level: cdshealpix.ring.healpix_to_lonlat(
                cell_ids, nside=2**level
            ),
        }
        converter = converters[self.indexing_scheme]

        lon, lat = converter(cell_ids, self.level)

        return np.asarray(lon.to("degree")), np.asarray(lat.to("degree"))

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
        from astropy.coordinates import Latitude, Longitude

        converters = {
            "nested": cdshealpix.nested.lonlat_to_healpix,
            "ring": lambda lon, lat, level: cdshealpix.ring.lonlat_to_healpix(
                lon, lat, nside=2**level
            ),
        }
        converter = converters[self.indexing_scheme]

        longitude = Longitude(lon, unit="degree")
        latitude = Latitude(lat, unit="degree")

        return converter(longitude, latitude, self.level)

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
        converters = {
            "nested": cdshealpix.nested.vertices,
            "ring": lambda cell_ids, level, **kwargs: cdshealpix.ring.vertices(
                cell_ids, nside=2**level, **kwargs
            ),
        }
        converter = converters[self.indexing_scheme]

        lon_, lat_ = converter(cell_ids, self.level, step=1)

        lon = np.asarray(lon_.to("degree"))
        lat = np.asarray(lat_.to("degree"))

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

        from healpix_geo.nested import zoom_to

        return zoom_to(cell_ids, self.level, level)


def construct_chunk_ranges(chunks, until):
    start = 0

    for chunksize in chunks:
        stop = start + chunksize
        if stop > until:
            stop = until
            if start == stop:
                break

        if until - start < chunksize:
            chunksize = until - start

        yield chunksize, slice(start, stop)
        start = stop


def subset_chunks(chunks, indexer):
    def _subset_slice(offset, chunk, indexer):
        if offset >= indexer.stop or offset + chunk < indexer.start:
            # outside slice
            return 0
        elif offset >= indexer.start and offset + chunk < indexer.stop:
            # full chunk
            return chunk
        else:
            # partial chunk
            left_trim = indexer.start - offset
            right_trim = offset + chunk - indexer.stop

            if left_trim < 0:
                left_trim = 0

            if right_trim < 0:
                right_trim = 0

            return chunk - left_trim - right_trim

    def _subset_array(offset, chunk, indexer):
        mask = (indexer >= offset) & (indexer < offset + chunk)

        return np.sum(mask.astype(int))

    def _subset(offset, chunk, indexer):
        if isinstance(indexer, slice):
            return _subset_slice(offset, chunk, indexer)
        else:
            return _subset_array(offset, chunk, indexer)

    if chunks is None:
        return None

    chunk_offsets = np.cumulative_sum(chunks, include_initial=True)
    total_length = chunk_offsets[-1]

    if isinstance(indexer, slice):
        indexer = slice(*indexer.indices(total_length))

    trimmed_chunks = tuple(
        _subset(offset, chunk, indexer)
        for offset, chunk in zip(chunk_offsets[:-1], chunks)
    )

    return tuple(int(chunk) for chunk in trimmed_chunks if chunk > 0)


def extract_chunk(index, slice_):
    return index.isel(slice_).cell_ids()


# optionally replaces the PandasIndex within HealpixIndex
class HealpixMocIndex(xr.Index):
    """More efficient index for healpix cell ids based on a MOC

    This uses the rust `moc crate <https://crates.io/crates/moc>`_ to represent
    cell ids as a set of disconnected ranges at level 29, vastly reducing the
    memory footprint and computation time of set-like operations.

    .. warning::

       Only supported for the ``nested`` scheme.

    See Also
    --------
    healpix_geo.nested.RangeMOCIndex
        The low-level implementation of the index functionality.
    """

    def __init__(self, index, *, dim, name, grid_info, chunksizes):
        self._index = index
        self._dim = dim
        self._grid_info = grid_info
        self._name = name
        self._chunksizes = chunksizes

    @property
    def size(self):
        """The number of indexed cells."""
        return self._index.size

    @property
    def nbytes(self):
        """The number of bytes occupied by the index.

        .. note::
           This does not take any (constant) overhead into account.
        """
        return self._index.nbytes

    @property
    def chunksizes(self):
        """The size of the chunks of the indexed coordinate."""
        return self._chunksizes

    @classmethod
    def from_array(cls, array, *, dim, name, grid_info):
        """Construct an index from a raw array.

        Parameters
        ----------
        array : array-like
            The array of cell ids as uint64. If the size is equal to the total
            number of cells at the given refinement level, creates a full domain
            index without looking at the cell ids. If a chunked array, it will
            create indexes for each chunk and then merge the chunk indexes
            in-memory.
        dim : hashable
            The dimension of the index.
        name : hashable
            The name of the indexed coordinate.
        grid_info : xdggs.HealpixInfo
            The grid parameters.

        Returns
        -------
        index : HealpixMocIndex
            The resulting index.
        """
        if grid_info.indexing_scheme != "nested":
            raise ValueError(
                "The MOC index currently only supports the 'nested' scheme"
            )

        if array.ndim != 1:
            raise ValueError("only 1D cell ids are supported")

        if array.size == 12 * 4**grid_info.level:
            index = RangeMOCIndex.full_domain(grid_info.level)
        elif isinstance(array, dask_array_type):
            from functools import reduce

            import dask

            [indexes] = dask.compute(
                dask.delayed(RangeMOCIndex.from_cell_ids)(grid_info.level, chunk)
                for chunk in array.astype("uint64").to_delayed()
            )
            index = reduce(RangeMOCIndex.union, indexes)
        else:
            index = RangeMOCIndex.from_cell_ids(grid_info.level, array.astype("uint64"))

        chunksizes = {dim: array.chunks[0] if hasattr(array, "chunks") else None}
        return cls(
            index, dim=dim, name=name, grid_info=grid_info, chunksizes=chunksizes
        )

    def _replace(self, index, chunksizes):
        return type(self)(
            index,
            dim=self._dim,
            name=self._name,
            grid_info=self._grid_info,
            chunksizes=chunksizes,
        )

    @classmethod
    def from_variables(cls, variables, *, options):
        """Create a new index object from the cell id coordinate variable

        Parameters
        ----------
        variables : dict-like
            Mapping of :py:class:`Variable` objects holding the coordinate labels
            to index.
        options : dict-like
            Mapping of arbitrary options to pass to the HealpixInfo object.

        Returns
        -------
        index : Index
            A new Index object.
        """
        name, var, dim = _extract_cell_id_variable(variables)
        grid_info = HealpixInfo.from_dict(var.attrs | options)

        return cls.from_array(var.data, dim=dim, name=name, grid_info=grid_info)

    def create_variables(self, variables):
        """Create new coordinate variables from this index

        Parameters
        ----------
        variables : dict-like, optional
            Mapping of :py:class:`Variable` objects.

        Returns
        -------
        index_variables : dict-like
            Dictionary of :py:class:`Variable` objects.
        """
        name = self._name
        if variables is not None and name in variables:
            var = variables[name]
            attrs = var.attrs
            encoding = var.encoding
        else:
            attrs = None
            encoding = None

        chunks = self._chunksizes[self._dim]
        if chunks is not None:
            import dask
            import dask.array as da

            chunk_arrays = [
                da.from_delayed(
                    dask.delayed(extract_chunk)(self._index, slice_),
                    shape=(chunksize,),
                    dtype="uint64",
                    name=f"chunk-{index}",
                    meta=np.array((), dtype="uint64"),
                )
                for index, (chunksize, slice_) in enumerate(
                    construct_chunk_ranges(chunks, self._index.size)
                )
            ]
            data = da.concatenate(chunk_arrays, axis=0)
            var = xr.Variable(self._dim, data, attrs=attrs, encoding=encoding)
        else:
            var = xr.Variable(
                self._dim, self._index.cell_ids(), attrs=attrs, encoding=encoding
            )

        return {name: var}

    def isel(self, indexers):
        """Subset the index using positional indexers.

        Parameters
        ----------
        indexers : dict-like
            A dictionary of positional indexers as passed from
            :py:meth:`Dataset.isel` and where the entries have been filtered for
            the current index.

        Returns
        -------
        maybe_index : Index
            A new Index object or ``None``.
        """
        indexer = indexers[self._dim]
        if isinstance(indexer, np.ndarray):
            if np.isdtype(indexer.dtype, "signed integer"):
                indexer = np.where(indexer >= 0, indexer, self.size + indexer).astype(
                    "uint64"
                )
            elif np.isdtype(indexer.dtype, "unsigned integer"):
                indexer = indexer.astype("uint64")
            else:
                raise ValueError("can only index with integer arrays or slices")

        new_chunksizes = {
            self._dim: subset_chunks(self._chunksizes[self._dim], indexer)
        }

        return self._replace(self._index.isel(indexer), chunksizes=new_chunksizes)

    def sel(self, labels: dict[Hashable, Any]) -> IndexSelResult:
        """Query the index using cell ids.

        Parameters
        ----------
        labels : dict-like of hashable to slice or array-like
            A dictionary of coordinate label indexers passed from
            :py:meth:`Dataset.sel` and where the entries have been filtered
            for the current index.

        Returns
        -------
        sel_results : :py:class:`IndexSelResult`
            An index query result object that contains dimension positional indexers.
            It may also contain new indexes, coordinate variables, etc.
        """
        indexer = labels[self._name]
        if isinstance(indexer, np.ndarray):
            if np.isdtype(indexer.dtype, "signed integer"):
                if np.any(indexer < 0):
                    raise ValueError("Cell ids can't be negative")

                indexer = np.astype(indexer, "uint64")
            elif np.isdtype(indexer.dtype, "unsigned integer"):
                indexer = np.astype(indexer, "uint64")
            else:
                raise ValueError("Can only index with cell id arrays or slices")

        dim_indexer, new_index = self._index.sel(indexer)
        new_chunksizes = {
            self._dim: subset_chunks(self._chunksizes[self._dim], dim_indexer)
        }

        return IndexSelResult(
            dim_indexers={self._dim: dim_indexer},
            indexes={self._name: self._replace(new_index, chunksizes=new_chunksizes)},
        )


@register_dggs("healpix")
class HealpixIndex(DGGSIndex):
    def __init__(
        self,
        cell_ids: Any | xr.Index,
        dim: str,
        grid_info: DGGSInfo,
        index_kind: str = "pandas",
    ):
        if not isinstance(grid_info, HealpixInfo):
            raise ValueError(f"grid info object has an invalid type: {type(grid_info)}")

        self._dim = dim

        if isinstance(cell_ids, xr.Index):
            self._index = cell_ids
        elif index_kind == "pandas":
            self._index = PandasIndex(cell_ids, dim)
        elif index_kind == "moc":
            self._index = HealpixMocIndex.from_array(
                cell_ids, dim=dim, grid_info=grid_info, name="cell_ids"
            )
        self._kind = index_kind

        self._grid = grid_info

    def values(self):
        if self._kind == "moc":
            return self._index._index.cell_ids()
        else:
            return self._index.index.values

    @classmethod
    def from_variables(
        cls: type["HealpixIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "HealpixIndex":
        _, var, dim = _extract_cell_id_variable(variables)

        index_kind = options.pop("index_kind", "pandas")

        grid_info = HealpixInfo.from_dict(var.attrs | options)

        return cls(var.data, dim, grid_info, index_kind=index_kind)

    def create_variables(self, variables):
        return self._index.create_variables(variables)

    def _replace(self, new_index: xr.Index):
        return type(self)(new_index, self._dim, self._grid, index_kind=self._kind)

    @property
    def grid_info(self) -> HealpixInfo:
        return self._grid

    def _repr_inline_(self, max_width: int):
        return f"HealpixIndex(level={self._grid.level}, indexing_scheme={self._grid.indexing_scheme}, kind={self._kind})"
