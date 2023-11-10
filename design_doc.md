# XDGGS - Design document

Xarrays extension for DGGS. Technical specifiactions.

## Goals

The goal of the `xddgs` library is to facilitate working with multiple Discrete Global Grid Systems (DGGSs) via a unified, high-level and user-friendly API that is deeply integrated with [Xarray](https://xarray.dev).

Examples of common DGGS features that `xdggs` should provide or facilitate:

- convert a DGGS from/to another grid (e.g., a DGGS, a latitude/longitude rectilinear grid, a raster grid, an unstructured mesh)
- convert a DGGS from/to vector data (points, lines, polygons)
- convert between different cell id representations of a same DGGS (e.g., uint64 vs. string)
- select data on a DGGS by cell ids or by geometries (spatial indexing)
- change DGGS resolution (upgrade or downgrade)
- re-organize cell ids (e.g., spatial shuffling / partitioning)
- plotting

Conversion between DGGS and other grids or vector features may requires specific interpolation or regridding methods.

`xdggs` should leverage the current recommended Xarray extension mechanisms ([apply_ufunc](https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html), [accessors](https://docs.xarray.dev/en/stable/internals/extending-xarray.html), [custom indexes](https://docs.xarray.dev/en/stable/internals/how-to-create-custom-index.html)) and possibly the future ones (e.g., variable encoders/decoders) to provide DGGS-specific functionality on top of Xarray core features.

`xdggs` should facitiltate interoperability with other existing Xarray extensions (e.g., [xvec](https://github.com/xarray-contrib/xvec) for vector data or [uxarray](https://github.com/UXARRAY/uxarray) for unstructured grids).

`xdggs` should also leverage the existing implementation of various DGGSs exposed to Python via 3rd-party libraries, here below referred as "backends". Preferrably, those backends would expose DGGS functionality in an efficient way as vectorized functions (e.g., working with NumPy arrays).

`xdggs` should try to follow standards and/or conventions defined for DGGS (see below) but MAY depart from them for practical reasons (e.g., common practices in popular DGGS libraries).

`xddgs` should also try to fulfill user needs in both GIS and Earth-System communities, which may each use DGGS in slightly different ways (see below).

When possible, `xddgs` operations should scale to fine resolutions (millions of cells) leveraging Xarray interoperability with Dask. This might not be always possible, though. Some operations (spatial indexing) may be hard to support at scale, which shouldn't be considered as high-priority.

## Non-Gloals

`xdggs` should focus on providing the core DGGS functionality and operations that are listed above. Higher-level operations that can be implemented by combining together those core operations are out-of-scope and should be implemented in downstream libraries.

`xddgs` should try not re-inventing the wheel and delegate to Xarray API when possible.

`xdggs` does not implement any particular DGGS from scratch. `xdggs` does not aim at providing _all the functionality provided by each grid_ (e.g., some functionality may be very specific to one DGGS and not supported by other DGGSs, or some functionality may not be available yet in one DGGS Python backend).

Although some DGGS may handle both the spatial and temporal domains in a joint fashion, `xdggs` focuses primarily on the spatial domain. The temporal domain is considered as orthogonal and already benefits from many core features provided by Xarray.

## Discrete Global Grid Systems

### Standards and Conventions

### Backends (Python)

## Representation of DGGS data in xdggs

`xdggs` represents a DGGS as an Xarray Dataset or DataArray containing a 1-dimensional coordinate with cell ids as labels and with grid name & parameters as attributes. This coordinate is indexed using a custom, Xarray-compatible `DGGSIndex`.

`xdggs` does not support a Dataset or DataArray with multiple coordinates indexed with a `DGGSIndex`.

The cell ids in the 1-dimensional coordinate are all relative to the _exact same_ grid (i.e., same system, same parameter values and same resolution). For simplicity, `xdggs` does not support mixed-resolutions cell ids in the same coordinate.

### DGGSIndex

`xdggs.DGGSIndex` is the base class for all Xarray DGGS-aware indexes. It inherits from `xarray.indexes.Index` and has the following specifications:

- It encapsulates an `xarray.indexes.PandasIndex` built from cell ids so that selection and alignment by cell id is possible
- It might also eventually encapsulate a spatial index (RTree, KDTree) to enable data selection by geometries, e.g., find nearest cell centroids, extract all cells intersecting a polygon, etc.
  - Alternatively, spatial indexing might be enabled by explicit conversion of cells to vector geometries and then by reusing the Xarray spatial indexes available in [xvec](https://github.com/xarray-contrib/xvec)
- It partially implements the Xarray Index API to enable DGGS-aware alignment and selection
  - Calls are most often redirected to the encapsulated `PandasIndex`
  - Some pre/post checks or processing may be done, e.g., to prevent the alignment of two indexes that are not on the exact same grid.
-

## Conversion

## Data Selection
