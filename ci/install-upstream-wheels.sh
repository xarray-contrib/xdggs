#!/usr/bin/env bash

# force-remove re-installed versions
micromamba remove -y --force \
    xarray \
    pandas \
    numpy \
    healpy
python -m pip uninstall -y h3ronpy

# build-deps for upstream-dev healpy
micromamba install -y cython setuptools setuptools-scm "maturin=1.2"
python -m pip install pykg-config

# install from scientific-python wheels
python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    pandas \
    xarray

# pyarrow (as a dependency for both `pandas` and `h3ronpy`)
python -m pip install \
       --extra-index-url https://pypi.fury.io/arrow-nightlies \
       --pre \
       --prefer-binary \
       pyarrow

# install from github
python -m pip install --no-deps --upgrade --no-build-isolation \
    git+https://github.com/nmandery/h3ronpy \
    git+https://github.com/healpy/healpy
