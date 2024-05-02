#!/usr/bin/env bash

if which micromamba; then
    conda=micromamba
else
    conda=mamba
fi

# force-remove re-installed versions
$conda remove -y --force \
    xarray \
    pandas \
    healpy
python -m pip uninstall -y h3ronpy

# install healpy build deps
$conda install healpix_cxx cython setuptools setuptools-scm "maturin=1.2"
python -m pip install pykg-config

# install from scientific-python wheels
python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --no-deps \
    --pre \
    --upgrade \
    pandas \
    numpy \
    xarray
# pyarrow nightly builds
python -m pip install \
    -i https://pypi.fury.io/arrow-nightlies/ \
    --prefer-binary \
    --no-deps \
    --pre \
    --upgrade \
    pyarrow


# install from github
python -m pip install --no-deps --upgrade \
    git+https://github.com/nmandery/h3ronpy
python -m pip install --no-deps --upgrade --no-build-isolation \
    git+https://github.com/healpy/healpy
