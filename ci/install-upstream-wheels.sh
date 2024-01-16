#!/usr/bin/env bash

# force-remove re-installed versions
micromamba remove -y --force \
    xarray \
    healpy
micromamba install -y cfitsio
python -m pip uninstall -y h3ronpy

# install from scientific-python wheels
python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    xarray

# install from github
python -m pip install --no-deps --upgrade \
    git+https://github.com/nmandery/h3ronpy

python -m pip install --no-deps --upgrade --no-build-isolation \
    git+https://github.com/healpy/healpy
