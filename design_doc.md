# XDGGS - Design document

Xarrays extension for DGGS. Technical specifications.

## Goals

The goal of the `xdggs` library is to facilitate working with multiple Discrete Global Grid Systems (DGGSs) via a unified, high-level and user-friendly API that is deeply integrated with [Xarray](https://xarray.dev).
This document describes the in-memory representation of DGGS data in Python environments.

Examples of common DGGS features that `xdggs` should provide or facilitate:

- convert a DGGS from/to another grid (e.g., a DGGS, a latitude/longitude rectilinear grid, a raster grid, an unstructured mesh)
- convert a DGGS from/to vector data (points, lines, polygons, envelopes)
- convert between different cell id representations of a same DGGS (e.g., uint64 vs. string)
- select data on a DGGS by cell ids or by geometries (spatial indexing)
- expand and reduce the available resolutions of a DGGS using down and upsampling, respectively.
- operations between similar DGGS (with auto-alignment)
- re-organize cell ids (e.g., spatial shuffling / partitioning)
- plotting

Conversion between DGGS and other grids or vector features may requires specific interpolation or regridding methods.

`xdggs` should leverage the current recommended Xarray extension mechanisms ([apply_ufunc](https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html), [accessors](https://docs.xarray.dev/en/stable/internals/extending-xarray.html), [custom indexes](https://docs.xarray.dev/en/stable/internals/how-to-create-custom-index.html)) and possibly the future ones (e.g., variable encoders/decoders) to provide DGGS-specific functionality on top of Xarray core features.

`xdggs` should facitiltate interoperability with other existing Xarray extensions (e.g., [xvec](https://github.com/xarray-contrib/xvec) for vector data or [uxarray](https://github.com/UXARRAY/uxarray) for unstructured grids).

`xdggs` should also leverage the existing implementation of various DGGSs exposed to Python via 3rd-party libraries, here below referred as "backends". Preferrably, those backends would expose DGGS functionality in an efficient way as vectorized functions (e.g., working with NumPy arrays).

`xdggs` should try to follow standards and/or conventions defined for DGGS (see below). However, we may need to depart from them for practical reasons (e.g., common practices in popular DGGS libraries that do not fit well with the proposed standards). Strict adherence to a standard is welcome but shouldn't be enforced by all means.

`xdggs` should also try to support applications in both GIS and Earth-System communities, which may each use DGGS in slightly different ways (see examples below).

When possible, `xdggs` operations should scale to fine DGGS resolutions (billions of cells). This can be done vertically using backends with vectorized bindings of DGGS implementations written in low-level languages and/or horizontally leveraging Xarray interoperability with Dask. Some operations like spatial indexing may be hard to scale horizontally, though. For the latter, we should probably focus `xdggs` development first towards good vertical scaling before figuring out how they can be scaled horizontally (for reference, see [dask-geopandas](https://github.com/geopandas/dask-geopandas) and [spatialpandas](https://github.com/holoviz/spatialpandas)).

## Non-Gloals

`xdggs` should focus on providing the core DGGS functionality and operations that are listed above. Higher-level operations that can be implemented by combining together those core operations are out-of-scope and should be implemented in downstream libraries. Likewise, there may be many ways of resampling a grid to a DGGS ; `xdggs` should support the most common methods but not try to support _all of them_.

`xdggs` should try not re-inventing the wheel and delegate to Xarray API when possible.

`xdggs` does not implement any particular DGGS from scratch. `xdggs` does not aim at providing _all the functionality provided by each grid_ (e.g., some functionality may be very specific to one DGGS and not supported by other DGGSs, or some functionality may not be available yet in one DGGS Python backend).

Although some DGGS may handle both the spatial and temporal domains in a joint fashion, `xdggs` focuses primarily on the spatial domain. The temporal domain is considered as orthogonal and already benefits from many core features provided by Xarray.

## Discrete Global Grid Systems

A Discrete Global Grid System (DGGS) can be roughly defined as a partitioning or tessellation of the entire Earth's surface into a finite number of "cells" or "zones". The shape and the properties of these cells generally vary from one DGGS to another. Most DGGSs are also hierarchical, i.e., the cells are arranged on recursively on multiple levels or resolutions. Follow the links in the subsection below for a more strict and detailed definition of a DGGS.

DGGSs may be used in various ways, e.g.,

- Applications in Earth-system modelling seem to use DGGS as grids of contiguous, fixed-resolution cells covering the entire Earth's surface or a region of interest (figure 1). This makes the analysis of simulation outputs on large extents of the Earth's surface easier. DGGS may also be used as pyramid data (multiple stacked datasets at different resolutions)
- Applications in GIS often consist of using DGGS to display aggregated (vector) data as a collection of cells with a more complex spatial distribution (sparse) and sometimes with mixed resolutions (figures 2 and 3).

![figure1](https://user-images.githubusercontent.com/4160723/281698490-31cb5ce8-64db-4bbf-a0d9-a8d6597bb2df.png)
Figure 1: DGGS data as contiguous cells of fixed resolution ([source](https://danlooo.github.io/DGGS.jl/))

![figure2](https://github.com/benbovy/xdggs/assets/4160723/430fd646-220a-4027-8212-1d927bb339ba)

Figure 2: Data aggreated on DGGS (H3) sparsely distributed cells of fixed resolution ([source](https://medium.com/@jesse.b.nestler/how-to-convert-h3-cell-boundaries-to-shapely-polygons-in-python-f7558add2f63)).

![image](https://github.com/benbovy/xdggs/assets/4160723/f2e4ec02-d88e-475e-9067-e93cf185923e)

Figure 3: Raster data converted as DGGS (H3) cells of mixed resolutions ([source](https://github.com/nmandery/h3ronpy)).

### Standards and Conventions

The [OGC abstract specification topic 21](http://www.opengis.net/doc/AS/dggs/2.0) defines properties of a DGGS including the reference systems of its grids.

However, there is no consensus yet about the actual specification on how to work with DGGD data.
[OGC API draft](https://github.com/opengeospatial/ogcapi-discrete-global-grid-systems) defines ways of how to access DGGS data.
The [DGGS data specification draft](https://github.com/danlooo/dggs-data-spec). aims to specify the storage format of DGGS data.

There are some discrepancies between the proposed standards and popular DGGS libraries (H3, S2, HealPIX). For example regarding the term used to define a grid unit: The two specifications above use "zone", S2/H3 use "cell" and HealPIX uses "pixel".
OGC abstract specification topic 21 defines the region as a zone and its boundary geometry as a cell.
Although in this document we use "cell", the term to choose for `xdggs` is still open for discussion.

### Backends (Python)

Several Python packages are currently available for handling certain DGGSs. They mostly consist of Python bindings of DGGS implementations written in C/C++/Rust. Here is a list (probably incomplete):

- [healpy](https://github.com/healpy/healpy): Python bindings of [HealPix](https://healpix.sourceforge.io/)
  - mostly vectorized
- [rhealpixdggs-py](https://github.com/manaakiwhenua/rhealpixdggs-py): Python/Numpy implementation of rHEALPix
- [h3-py](https://github.com/uber/h3-py): "official" Python bindings of [H3](https://h3geo.org/)
  - experimental and incomplete vectorized version of H3's API (removed in the forthcoming v4 release?)
- [h3pandas](https://github.com/DahnJ/H3-Pandas): integration of h3-py (non-vectorized) with pandas and geopandas
- [h3ronpy](https://github.com/nmandery/h3ronpy): Python bindings of [h3o](https://github.com/HydroniumLabs/h3o) (Rust implementation of H3)
  - provides high-level features (conversion, etc.) working with arrow, numpy (?), pandas/geopandas and polars
- [s2geometry](https://github.com/google/s2geometry): Python bindings generated with SWIG
  - not vectorized nor very "pythonic"
  - plans to switch to pybind11 (no time frame given)
- [spherely](https://github.com/benbovy/spherely): Python bindings of S2, mostly copying shapely's API
  - provides numpy-like universal functions
  - not yet ready for use
- [dggrid4py](https://github.com/allixender/dggrid4py): Python wrapper for [DGGRID](https://github.com/sahrk/DGGRID)
  - DGGRID implements many DGGS variants!
  - DGGRID current design makes it hardly reusable from within Python in an optimal way (the dggrid wrapper communicates with DGGRID through OS processes and I/O generated files)

## Representation of DGGS Data in Xdggs

`xdggs` represents a DGGS as an Xarray Dataset or DataArray containing a 1-dimensional coordinate with cell ids as labels and with grid name, resolution & parameters (optional) as attributes. This coordinate is indexed using a custom, Xarray-compatible `DGGSIndex`. Multiple dimensions may be used if the coordinate consists of multiple parts, e.g., polyhedron face, x, and y on that face in DGGRID PROJTRI.

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

Each DGGS supported in `xdggs` has its own subclass of `DGGSIndex`, e.g.,

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

## Conversion from/to DGGS

DGGS data may be created from various sources, e.g.,

- regridded from a latitude/longitude rectilinear grid
- regridded from an unstructured grid
- regridded and reprojected from a raster having a local projection
- aggregated from vector point data
- filled from polygon data

Conversely, DGGS data may be converted to various forms, e.g.,

- regridded on a latitude/longitude rectilinear grid
- rasterized (resampling / projection)
- conversion to vector point data (cell centroids)
- conversion to vector polygon data (cell boundaries)

Here is a tentative API based on Dataset/DataArray `.dggs` accessors (note: other options are discussed in [this issue](https://github.com/xarray-contrib/xdggs/issues/13)):

```python
# "convert" directly from existing cell ids coordinate to DGGS
# basically an alias to ds.set_xindex(..., DGGSIndex)
ds.dggs.from_cell_ids(...)

# convert from lat/lon grid
ds.dggs.from_latlon_grid(...)

# convert from raster
ds.dggs.from_raster(...)

# convert from point data
ds.dggs.from_points(...)

# convert from point data (with aggregation)
ds.dggs.from_points_aggregate(...)

# convert from point data (with aggregation using Xarray API)
ds.dggs.from_points(...).groupby(...).mean()

# convert to lat/lon grid
ds.dggs.to_latlon_grid(...)

# convert to raster
ds.dggs.to_raster(...)

# convert to points (cell centroids)
ds.dggs.to_points(...)

# convert to polygons (cell boundaries)
ds.dggs.to_polygons(...)
```

In the API methods above, the "dggs" accessor name serves as a prefix.

Those methods are all called from an existing xarray Dataset (DataArray) and should all return another Dataset (DataArray):

- Xarray has built-in support for regular grids
- for rasters, we could return objects that are [rioxarray](https://github.com/corteva/rioxarray)-friendly
- for vector data, we could return objects that are [xvec](https://github.com/xarray-contrib/xvec)-friendly (coordinate of shapely objects)
- etc.

## Extracting DGGS Cell Geometries

DGGS cell geometries could be extracted using the conversion methods proposed above. Alternatively, it would be convenient to get them directly as xarray DataArrays so that we can for example manually assign them as coordinates.

The API may look like:

```python
# return a DataArray of DGGS cell centroids as shapely.POINT objects
ds.dggs.cell_centroids()

# return two DataArrays of DGGS cell centroids as lat/lon coordinates
ds.dggs.cell_centroids_coords()

# return a DataArray of DGGS cell boundaries as shapely.POLYGON objects
ds.dggs.cell_boundaries()

# return a DataArray of DGGS cell envelopes as shapely.POLYGON objects
ds.dggs.cell_envelopes()
```

## Indexing and Selecting DGGS Data

### Selection by Cell IDs

The simplest way to select DGGS data is by cell ids. This can be done directly using Xarray's API (`.sel()`):

```python
ds.sel(cell=value)
```

where `value` can be a single cell id (integer or string/token?) or an array-like of cell ids. This is easily supported by the DGGSIndex encapsulating a PandasIndex. We might also want to support other `value` types, e.g.,

- assuming that DGGS cell ids are defined such that contiguous cells in space have contiguous id values, we could provide a `slice` to define a range of cell ids.
- DGGSIndex might implement some DGGS-aware logic such that it auto-detects if the given input cells are parent cells (lower DGGS resolution) and then selects all child cells accordingly.

We might want to select cell neighbors (i.e., return a new Dataset/DataArray with a new neighbor dimension), probably via a specific API (`.dggs` accessors).

### Selection by Geometries (Spatial Indexing)

Another useful way of selecting DGGS data is from input geometries (spatial queries), e.g.,

- Select all cells that are the closest to a collection of data points
- Select all cells that intersects with or are fully contained in a polygon

This kind of selection requires spatial indexes as this can not be done with a pandas index (see [this issue](https://github.com/xarray-contrib/xdggs/issues/16)).

If we support spatial indexing directly in `xdggs`, we can hardly reuse Xarray's `.sel()` for spatial queries as `ds.sel(cell=shapely.Polygon(...))` would look quite confusing. Perhaps better would be to align with [xvec](https://github.com/xarray-contrib/xvec) and have a separate `.dggs.query()` method.

Alternatively, we could just get away with the conversion and cell geometry extraction methods proposed above and leave spatial indexes/queries to [xvec](https://github.com/xarray-contrib/xvec).

## Handling hierarchical DGGS

DGGS are grid systems with grids of the same topology but different spatial resolution.
There is a hierarchical relationship between grids of different resolutions.
Even though the coordinate of one grid in the DGGS of a Dataset (DataArray) is limited to cell ids of same resolution (no mixed-resolutions), `xdggs` can still provide functionality to deal with the hierarchical aspect of DGGSs.

Selection by parent cell ids may be in example (see section above). Another example would be to have utility methods to explicitly change the grid resolution (see [issue #18](https://github.com/xarray-contrib/xdggs/issues/18) for more details and discussion).
One can also store DGGS data at all resolutions as a list of datasets.

However, like in hexagonal grids of aperture 3 or 4 (e.g. DGGRID ISEA4H), the parent child relationship can be also ambiguous.
The actual spatial aggregation functions in the subclasses might be implemented differently depending on the selected DGGS.

## Operations between similar DGGS (alignment)

Computation involving multiple DGGS datasets (or dataarrays) often requires to align them together. Sometimes this can be trivial (same DGGS with same resolution and parameter values) but in other cases this can be complex (requires regridding or a change of DGGS resolution).

In Xarray, alignment of datasets (dataarrays) is done primarily via their indexes. Since a DGGSIndex wraps a PandasIndex, it is easy to support alignment by cells ids (trivial case). At the very least, a DGGSIndex should raise an error when trying to align cell ids that do not refer to the exact same DGGS (i.e., same system, resolution and parameter values). For the complex cases, it may be preferable to handle them manually instead of trying to make the DGGSIndex perform the alignment automatically. Regridding and/or changing the resolution of a DGGS (+ data aggregation) often highly depend on the use-case so it might be hard to find a default behavior. Also performing those operations automatically and implicitly would probably feel too magical. That being said, in order to help alignment `xdggs` may provide some utility methods to change the grid resolution (see section above) and/or to convert from one DGGS to another.

## Plotting

Three approaches are possible (non-mutually exclusive):

1. convert cell data into gridded or raster data (choose grid/raster resolution depending on the resolution of the rendered figure) and then reuse existing python plotting libraries (matplotlib, cartopy) maybe through xarray plotting API
2. convert cell data into vector data and plot the latter via, e.g., [xvec](https://github.com/xarray-contrib/xvec) or [geopandas](https://github.com/geopandas/geopandas) API
3. leverage libraries that support plotting DGGS data, e.g., [lonboard](https://github.com/developmentseed/lonboard) enables interactive plotting in Jupyter via deck.gl, which has support of H3 and S2 cell data.

The 3rd approach (lonboard) is efficient for plotting large DGGS data: we would only need to transfer cell ids (tokens) and cell data and then let deck.gl render the cells efficiently in the web browser using the GPU. For approach 1, we might want to investigate using [datashader](https://github.com/holoviz/datashader) to set both the resolution and raster extent dynamically. Likewise for approach 2, we could dynamically downgrade the DGGS resolution and aggregate the data before converting it into vector data in order to allow (interactive) plotting of large DGGS data.
