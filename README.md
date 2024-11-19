# xdggs

`xdggs` is an open-source Python package that provides tools for handling geospatial data using Discrete Global Grid Systems (DGGS).

It enables efficient manipulation and analysis of multi-dimensional gridded data within a DGGS framework, supporting spatial data processing, resampling, and aggregation on both global and regional scales.

Inspired by the growing need for scalable geospatial data analysis with DGGS, `xdggs` is built upon the robust [Xarray](https://xarray.pydata.org/) ecosystem, which simplifies working with labeled multi-dimensional arrays.

As an extension of Xarray, `xdggs` leverages Xarray's capabilities, including seamless access to formats like [NetCDF](https://www.unidata.ucar.edu/software/netcdf/), [Zarr](https://zarr.readthedocs.io/), and parallelization through [Dask](https://www.dask.org/), to provide a powerful and flexible toolkit for geospatial analysis.

## Key Features

- **Seamless Integration with Xarray**: Use `xdggs` alongside Xarray's powerful tools for managing labeled, multi-dimensional data.
- **Support for DGGS**: Convert geospatial data into DGGS representations, allowing for uniform spatial partitioning of the Earth's surface.
- **Spatial Resampling**: Resample data on DGGS grids, enabling downscaling or upscaling across multiple resolutions.
- **DGGS Aggregation**: Perform spatial aggregation of data on DGGS cells.
- **Efficient Data Management**: Manage large datasets with Xarray's lazy loading, Dask integration, and chunking to optimize performance.

## Documentation

You can find the documentation in [https://xdggs.readthedocs.io/en/latest/](https://xdggs.readthedocs.io/en/latest/).

## Demo

![xdggs demo](https://raw.githubusercontent.com/xarray-contrib/xdggs/refs/heads/main/xdggs-cropped.gif)

## Getting Started

As an example, this is how you would use `xdggs` to reconstruct geographical coordinates from the cell ids then create an interactive plot indicating cell ids, data values and the associated geographical coordinates:

```python
import xarray as xr
import xdggs

# Load the dataset created by ./examples/prepare_dataset_h3.ipynb
ds = xr.open_dataset("data/h3.nc", engine="netcdf4")

# Decode DGGS coordinates
ds_idx = ds.pipe(xdggs.decode)

# Assign geographical coordinates
ds_idx = ds_idx.dggs.assign_latlon_coords()

# Interactive visualization
ds_idx['air'].isel(time=0).compute().dggs.explore(center=0, cmap="viridis", alpha=0.5)

```

## Dependencies

- Python >= 3.10
- Xarray >= 2023.09.0
- NumPy >= 1.24.0
- Dask >= 2023.10.0 (optional, for parallel computing)

## Roadmap

We have exciting plans to expand xdggs with new features and improvements. You can check out our roadmap in the [design_doc.md](https://github.com/xarray-contrib/xdggs/blob/main/design_doc.md) file for details on the design of xdggs, upcoming features, and future enhancements.

## Contributing

We welcome contributions to `xdggs`! Please follow these steps to get involved:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and write tests.
4. Ensure all tests pass (`pytest`).
5. Submit a pull request!

## License

`xdggs` is licensed under the Apache License License. See [LICENSE](https://github.com/xarray-contrib/xdggs/blob/main/LICENSE) for more details.

## Acknowledgments
