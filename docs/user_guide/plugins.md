# Ecosystem & Plugins

`xdggs` ships with built-in support for the H3 and HEALPix(+geo) grid systems. Support for additional discrete global grid systems (DGGS) is possible through community-maintained plugin packages that build on top of `xdggs`.

## xdggs-dggrid4py

[`xdggs-dggrid4py`](https://github.com/LandscapeGeoinformatics/xdggs-dggrid4py) is an `xdggs` plugin that wraps [`dggrid4py`](https://github.com/LandscapeGeoinformatics/dggrid4py) / [DGGRID](https://github.com/sahrk/DGGRID) to enable spatial operations and xarray indexing on IGEO7 and potentially other DGGS grids supported by DGGRID.

Check out the docs:

- [An xdggs plugin for IGEO7 DGGS](https://xdggs-dggrid4py.readthedocs.io/en/latest/)
- [Tutorial and demo incl. simple dataset conversions](https://xdggs-dggrid4py.readthedocs.io/en/latest/tutorials/demo_xdggs-dggrid4py.html)

## xdggs-dggal

[`xdggs-dggal`](https://github.com/LandscapeGeoinformatics/xdggs-dggal) is a companion plugin adding support for DGGS grids provided by [DGGAL](https://dggal.org/) (WIP).


## pydggsapi

[`pydggsapi`](https://github.com/LandscapeGeoinformatics/pydggsapi/) is a python FastAPI implementation for the OGC DGGS API standard. It can use `xdggs` for [Zarr-DGGS](https://github.com/zarr-conventions/dggs) datasets

