# XDGGS - Design document

Xarrays extension for DGGS. Technical specifiactions.

## Goals

The goal of the `xddgs` library is to facilitate working with multiple Discrete Global Grid Systems (DGGSs) via a unified, high-level and user-friendly API that is deeply integrated with [Xarray](https://xarray.dev).

Examples of common DGGS features that `xdggs` should provide or facilitate:

- convert a DGGS from/to another grid (e.g., a DGGS, a latitude/longitude rectilinear grid, a raster grid, an unstructured mesh)
- convert a DGGS from/to vector data (points, lines, polygons, envelopes)
- convert between different cell id representations of a same DGGS (e.g., uint64 vs. string)
- select data on a DGGS by cell ids or by geometries (spatial indexing)
- change DGGS resolution (upgrade or downgrade)
- re-organize cell ids (e.g., spatial shuffling / partitioning)
- plotting

Conversion between DGGS and other grids or vector features may requires specific interpolation or regridding methods.

`xdggs` should leverage the current recommended Xarray extension mechanisms ([apply_ufunc](https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html), [accessors](https://docs.xarray.dev/en/stable/internals/extending-xarray.html), [custom indexes](https://docs.xarray.dev/en/stable/internals/how-to-create-custom-index.html)) and possibly the future ones (e.g., variable encoders/decoders) to provide DGGS-specific functionality on top of Xarray core features.

`xdggs` should facitiltate interoperability with other existing Xarray extensions (e.g., [xvec](https://github.com/xarray-contrib/xvec) for vector data or [uxarray](https://github.com/UXARRAY/uxarray) for unstructured grids).

`xdggs` should also leverage the existing implementation of various DGGSs exposed to Python via 3rd-party libraries, here below referred as "backends". Preferrably, those backends would expose DGGS functionality in an efficient way as vectorized functions (e.g., working with NumPy arrays).

`xdggs` should try to follow standards and/or conventions defined for DGGS (see below). However, we may need to depart from them for practical reasons (e.g., common practices in popular DGGS libraries that do not fit well with the proposed standards). Strict adherence to a standard is welcome but shouldn't be enforced by all means.

`xddgs` should also try to fulfill user needs in both GIS and Earth-System communities, which may each use DGGS in slightly different ways (see below).

When possible, `xddgs` operations should scale to fine resolutions (millions of cells) leveraging Xarray interoperability with Dask. This might not be always possible, though. Some operations (spatial indexing) may be hard to support at scale and it shouldn't be a high development priority.

## Non-Gloals

`xdggs` should focus on providing the core DGGS functionality and operations that are listed above. Higher-level operations that can be implemented by combining together those core operations are out-of-scope and should be implemented in downstream libraries. Likewise, there may be many ways of resampling a grid to a DGGS ; `xdggs` should support the most common methods but not try to support _all of them_.

`xddgs` should try not re-inventing the wheel and delegate to Xarray API when possible.

`xdggs` does not implement any particular DGGS from scratch. `xdggs` does not aim at providing _all the functionality provided by each grid_ (e.g., some functionality may be very specific to one DGGS and not supported by other DGGSs, or some functionality may not be available yet in one DGGS Python backend).

Although some DGGS may handle both the spatial and temporal domains in a joint fashion, `xdggs` focuses primarily on the spatial domain. The temporal domain is considered as orthogonal and already benefits from many core features provided by Xarray.

## Discrete Global Grid Systems

### Standards and Conventions

### Backends (Python)

## Representation of DGGS data in xdggs

`xdggs` represents a DGGS as an Xarray Dataset or DataArray containing a 1-dimensional coordinate with cell ids as labels and with grid name, resolution & parameters (optional) as attributes. This coordinate is indexed using a custom, Xarray-compatible `DGGSIndex`.

`xdggs` does not support a Dataset or DataArray with multiple coordinates indexed with a `DGGSIndex` (only one DGGS per object is supported).

The cell ids in the 1-dimensional coordinate are all relative to the _exact same_ grid, i.e., same grid system, same grid parameter values and same grid resolution! For simplicity, `xdggs` does not support cell ids of mixed-resolutions in the same coordinate.

### DGGSIndex

`xdggs.DGGSIndex` is the base class for all Xarray DGGS-aware indexes. It inherits from `xarray.indexes.Index` and has the following specifications:

- It encapsulates an `xarray.indexes.PandasIndex` built from cell ids so that selection and alignment by cell id is possible
- It might also eventually encapsulate a spatial index (RTree, KDTree) to enable data selection by geometries, e.g., find nearest cell centroids, extract all cells intersecting a polygon, etc.
  - Alternatively, spatial indexing might be enabled by explicit conversion of cells to vector geometries and then by reusing the Xarray spatial indexes available in [xvec](https://github.com/xarray-contrib/xvec)
- It partially implements the Xarray Index API to enable DGGS-aware alignment and selection
  - Calls are most often redirected to the encapsulated `PandasIndex`
  - Some pre/post checks or processing may be done, e.g., to prevent the alignment of two indexes that are not on the exact same grid.
- The `DGGSIndex.__init__()` constructor only requires cell ids and the name of the cell (array) dimension
- The `DGGSIndex.from_variables()` factory method parses the attributes of the given cell ids coordinates and creates the right index object (subclass) accordingly
- It declares a few abstract methods for grid-aware operations (e.g., convert between cell id and lat/lon point or geometry, etc.)
  - They can be implemented in subclasses (see below)
  - They are either called from within the DGGSIndex or from the `.dggs` Dataset/DataArray accessors

Each DGGS supported in `xddgs` has its own subclass of `DGGSIndex`, e.g.,

- `HealpixIndex` for Healpix
- `H3Index` for H3
- ...

A DGGSIndex can be set directly from a cell ids coordinate using the Xarray API:

```python
import xarray as xr
import xdggs

ds = xr.Dataset(
    coords={"cell": ("cell", [...], {"grid_name": "h3", "resolution": 3})}
)

# auto-detect grid system and parameters
ds.set_xindex("cell", xdggs.DGGSIndex)

# set the grid system and parameters manually
ds.set_xindex("cell", xdggs.H3Index, resolution=3)
```

The DGGSIndex is set automatically when converting a gridded or vector dataset to a DGGS dataset (see below).

## Conversion

## Data Selection

## Plotting

Three approaches are possible (non-mutually exclusive):

1. convert cell data into gridded or raster data (choose grid/raster resolution depending on the resolution of the rendered figure) and then reuse existing python plotting libraries (matplotlib, cartopy) maybe through xarray plotting API
2. convert cell data into vector data and plot the latter via, e.g., [xvec](https://github.com/xarray-contrib/xvec) or [geopandas](https://github.com/geopandas/geopandas) API
3. leverage libraries that support plotting DGGS data, e.g., [lonboard](https://github.com/developmentseed/lonboard) enables interactive plotting in Jupyter via deck.gl, which has support of H3 and S2 cell data.

The first and last approaches may be efficient in plotting large DGGS data. For approach 1, we might want to investigate using [datashader](https://github.com/holoviz/datashader) to set both the resolution and raster extent dynamically. For approach 3 (lonboard), we would only need to transfer cell ids (tokens) and cell data and then let deck.gl render the cells efficiently in the web browser using the GPU.

Although the second approach may not scale as best as the other ones, it is versatile and may produce nice looking graphics.
