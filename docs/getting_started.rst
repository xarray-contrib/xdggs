Quickstart
==========

``xdggs`` allows you to handle geospatial data using Discrete Global Grid Systems (DGGS).

You can install xdggs with pip:

.. code:: shell

   pip install xdggs


To use the package, import ``xdggs``. Below is a simple example to use xdggs

.. code:: python

    import xarray as xr
    import xdggs

    # Load the dataset created by ./examples/prepare_dataset_h3.ipynb
    ds = xr.open_dataset("data/h3.nc", engine="netcdf4")

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
