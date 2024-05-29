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
    git+https://github.com/Unidata/cftime \
    git+https://github.com/astropy/astropy \
    git+https://github.com/nmandery/h3ronpy

# install healpy from github
# need to run `auditwheel` to include the shared libs
mkdir -p built_wheel repaired_wheel
python -m pip wheel --no-deps git+https://github.com/healpy/healpy --wheel-dir built_wheel
auditwheel repair --plat linux_x86_64 -w repaired_wheel built_wheel/healpy-*.whl
python -m pip install --upgrade --no-deps repaired_wheel/healpy-*.whl
