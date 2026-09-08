from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import xarray as xr
from healpix_geo.nested import RangeMOCIndex
from xarray.core.indexes import IndexSelResult

from xdggs.healpix.grid_info import HealpixInfo
from xdggs.healpix.indexing_adapters import MocRangesIndexingAdapter
from xdggs.utils import _extract_cell_id_variable

try:
    import dask.array as da

    dask_array_type = (da.Array,)
except ImportError:
    dask_array_type = ()


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

        ellipsoid = grid_info.ellipsoid
        if ellipsoid is None:
            import healpix_geo.ellipsoid

            ellipsoid = healpix_geo.ellipsoid.resolve("sphere")

        if array.size == 12 * 4**grid_info.level:
            index = RangeMOCIndex.full_domain(grid_info.level)
        elif isinstance(array, dask_array_type):
            from functools import reduce

            import dask

            [indexes] = dask.compute(
                dask.delayed(RangeMOCIndex.from_cell_ids)(
                    grid_info.level, chunk, ellipsoid=ellipsoid
                )
                for chunk in array.astype("uint64").to_delayed()
            )
            index = reduce(RangeMOCIndex.union, indexes)
        else:
            index = RangeMOCIndex.from_cell_ids(
                grid_info.level, array.astype("uint64"), ellipsoid=ellipsoid
            )

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

    def create_variables(
        self, variables: Mapping[Any, xr.Variable] | None = None
    ) -> dict[Hashable, xr.Variable]:
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

        data = MocRangesIndexingAdapter(
            self._index, self._grid_info, self._name, dims=(self._dim,)
        )
        var = xr.Variable(self._dim, data, attrs=attrs, encoding=encoding)

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

    def sel(self, labels: dict[Hashable, Any], method: str = None) -> IndexSelResult:
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
