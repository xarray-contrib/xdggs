# xdggs

`xdggs` is an extension for [Xarray](https://xarray.pydata.org/) that provides tools to handle geospatial data using Discrete Global Grid Systems (DGGS). It allows efficient manipulation and analysis of multi-dimensional gridded data within a DGGS framework, facilitating spatial data processing, resampling, and aggregations on both global and regional scales.

## Key Features

- **Seamless Integration with Xarray**: Use `xdggs` alongside Xarray's powerful tools for managing labeled, multi-dimensional data.
- **Support for DGGS**: Convert geospatial data into DGGS representations, allowing for uniform spatial partitioning of the Earth's surface.
- **Spatial Resampling**: Resample data on DGGS grids, enabling downscaling or upscaling across multiple resolutions.
- **DGGS Aggregation**: Perform spatial aggregation of data on DGGS cells.
- **Efficient Data Management**: Manage large datasets with Xarray's lazy loading, Dask integration, and chunking to optimize performance.

## Installation

To install `xdggs`, you can clone the repository and install it using pip:

```bash
git clone https://github.com/your-repo/xdggs.git
cd xdggs
pip install .
```

Alternatively, you can install it directly via pip (once it's available on PyPI):

```bash
pip install xdggs
```

## Getting Started

Here's a quick example of how to use `xdggs` to convert your Xarray data to a DGGS grid and perform resampling.

```python
import xarray as xr
import xdggs

# Load your Xarray dataset
ds = xr.open_dataset('path_to_your_data.nc')

...

# Save the processed data
xarray_data.to_netcdf('resampled_data.nc')
```

## Dependencies

- Python >= 3.8
- Xarray >= 0.19
- NumPy >= 1.19
- Dask (optional, for parallel computing)

You can install the dependencies with:

```bash
pip install xarray numpy dask 
```

## Documentation

Under construction.

## Contributing

We welcome contributions to `xdggs`! Please follow these steps to get involved:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and write tests.
4. Ensure all tests pass (`pytest`).
5. Submit a pull request!

## License

`xdggs` is licensed under the Apache License License. See [LICENSE](LICENSE) for more details.

## Acknowledgments

This project was inspired by the increasing need for scalable geospatial data analysis using DGGS and is built upon the robust ecosystem of Xarray.
