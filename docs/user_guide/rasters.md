# Rasters and static plots

`xdggs` can resample DGGS data onto a pixel grid, either to write it to a raster file (like a GeoTIFF) or to draw it on a static `matplotlib` / `cartopy` map. The grid geometry comes from an optional rendering backend chosen by the grid type:

- **HEALPix**: [`healpix-plot`](https://github.com/GRID4EARTH/healpix-plot), installed with `pip install "xdggs[plot,raster]"`
- **H3**: not supported yet

`xdggs` itself does not depend on `rasterio`: {py:meth}`DataArray.dggs.to_raster <xarray.DataArray.dggs.to_raster>` returns an {py:class}`xarray.DataArray` with `y` / `x` dimensions, a CRS and an affine transform, which [`rioxarray`](https://corteva.github.io/rioxarray) can write.

## Sampling grids

The `sampling_grid` parameter selects the pixels:

- `"native"` (the default for `to_raster`): exactly one pixel per cell in the grid's native projection. For HEALPix, this is `+proj=healpix`, where every cell is a square rotated by 45°. The conversion is lossless and can be reversed with {py:func}`xdggs.from_raster`.
- a dict such as `{"shape": (720, 360)}`: a regular longitude / latitude grid. This is what most tools expect, but cells are duplicated or skipped depending on the resolution.
- a sampling grid object of the backend, for example `healpix_plot.AffineSamplingGrid(transform, shape, crs)` to sample on the pixels of any projected raster.

## Writing a GeoTIFF

```python
import rioxarray  # registers the `rio` accessor
import xdggs

ds = xdggs.tutorial.open_dataset("air_temperature", "healpix").load().dggs.decode()
arr = ds["air"].isel(time=0)

raster = arr.dggs.to_raster()  # lossless, in +proj=healpix
raster.rio.to_raster("air.tif")
```

Additional dimensions (like `time`) are kept, so `ds["air"].dggs.to_raster()` returns a `(time, y, x)` array that is written as a multi-band raster.

GeoTIFF has no GeoKeys for the HEALPix projection. GDAL therefore keeps the CRS in an `air.tif.aux.xml` side-car file, and `healpix-plot` also stores it as WKT in the `healpix_crs_wkt` TIFF tag, together with the grid parameters (`healpix_depth`, `healpix_indexing_scheme`, `healpix_ellipsoid`). Tools that do not use GDAL / PROJ may not recognize the CRS. There is no EPSG code for the ellipsoidal HEALPix projection yet.

## Reading it back

```python
raster = rioxarray.open_rasterio("air.tif").squeeze("band", drop=True)
cells = xdggs.from_raster(raster, arr.dggs.grid_info).dropna("cells")
```

Pixels of the bounding box that were not in the original data come back as `NaN`, hence the `dropna` (which also drops cells whose value was `NaN`). `from_raster` refuses rasters whose pixels are smaller than the cells (several pixel centres per cell), because that would need an aggregation.

## Static plots

```python
arr.dggs.plot(projection="Robinson", cmap="RdBu_r", colorbar=True)
arr.dggs.plot(projection="HEALPix")  # draws the cells themselves
```

`projection` is the name of a `cartopy` projection, a `cartopy.crs.Projection` object, or `"HEALPix"`. For the HEALPix projection the cells are drawn exactly (one pixel per cell); for the others the data is first resampled onto a longitude / latitude image. Plotting requires a 1D `DataArray`: select the other dimensions first.

## Known limitations (HEALPix)

- The native grid requires `level >= 1`: at level 0 the polar base cells are offset from the equatorial ones by half a cell.
- The HEALPix projection is interrupted. Values are not interpolated across facet boundaries, and longitude 180° belongs to the eastern edge only, which shows as a half-pixel sawtooth on the western edge of global maps.
- Only nearest-neighbour sampling is available.
