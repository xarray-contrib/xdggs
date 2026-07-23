# Ecosystem & Plugins

`xdggs` ships with built-in support for the H3 and HEALPix(+geo) grid systems. Support for additional discrete global grid systems (DGGS) is possible through community-maintained plugin packages that build on top of `xdggs`.

## Awesome HEALPix

[`pangeo-data/awesome-HEALPix`](https://github.com/pangeo-data/awesome-HEALPix) is a curated list of awesome tools and libraries related to HEALPix, the Hierarchical Equal Area isoLatitude Pixelization of the sphere, one of the currently most supported DGGS' in `xdggs`. Originally developed for cosmological studies and the analysis of the Cosmic Microwave Background (CMB), HEALPix has since expanded into geoinformatics and Earth observation, facilitating scalable spatial data analysis (such as in [GRID4EARTH](https://grid4earth.eu/)).

## xdggs-dggrid4py

[`xdggs-dggrid4py`](https://github.com/LandscapeGeoinformatics/xdggs-dggrid4py) is an `xdggs` plugin that wraps [`dggrid4py`](https://github.com/LandscapeGeoinformatics/dggrid4py) / [DGGRID](https://github.com/sahrk/DGGRID) to enable spatial operations and xarray indexing on IGEO7 and potentially other DGGS grids supported by DGGRID.

Check out the docs:

- [An xdggs plugin for IGEO7 DGGS](https://xdggs-dggrid4py.readthedocs.io/en/latest/)
- [Tutorial and demo incl. simple dataset conversions](https://xdggs-dggrid4py.readthedocs.io/en/latest/tutorials/demo_xdggs-dggrid4py.html)

## xdggs-dggal

[`xdggs-dggal`](https://github.com/LandscapeGeoinformatics/xdggs-dggal) is a companion plugin adding support for DGGS grids provided by [DGGAL](https://dggal.org/) (WIP).

## pydggsapi

[`pydggsapi`](https://github.com/LandscapeGeoinformatics/pydggsapi/) is a python FastAPI implementation for the OGC DGGS API standard. It can use `xdggs` for [Zarr-DGGS](https://github.com/zarr-conventions/dggs) datasets

## Awesome Discrete Global Grid Systems (DGGS)

[`awesome-discrete-global-grid-systems`](https://github.com/LandscapeGeoinformatics/awesome-discrete-global-grid-systems) is a more general collection of DGGS resources for science and standards, and fun and profit.
