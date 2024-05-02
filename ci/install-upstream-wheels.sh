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

# install from scientific-python wheels
python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --no-deps \
    --pre \
    --upgrade \
    pandas \
    xarray

# install from github
python -m pip install --no-deps --upgrade \
    git+https://github.com/nmandery/h3ronpy \
    git+https://github.com/healpy/healpy
