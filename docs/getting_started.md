# Quickstart

`xdggs` allows you to handle geospatial data using Discrete Global Grid Systems (DGGS).

You can install xdggs with pip:

```shell
pip install xdggs
```

To use the package, import `xdggs`. Below is a simple example to use xdggs:

```python
import xarray as xr
import xdggs

# Load the tutorial dataset
ds = xr.tutorial.open_dataset("air_temperature", "h3")

# Decode DGGS coordinates
ds_idx = ds.pipe(xdggs.decode)

# Assign geographical coordinates
ds_idx = ds_idx.dggs.assign_latlon_coords()

# Interactive visualization
ds_idx["air"].isel(time=0).compute().dggs.explore(
center=0, cmap="viridis", alpha=0.5
)

import xarray_regrid
import xarray

ds = xr.open_dataset("input_data.nc")
ds_grid = xr.open_dataset("target_grid.nc")

ds = ds.regrid.linear(ds_grid)

# or, for example:
ds = ds.regrid.conservative(ds_grid, latitude_coord="lat")
```
