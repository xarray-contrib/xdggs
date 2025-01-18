# Installation

`xdggs` can be installed from PyPI:

```shell
pip install xdggs
```

from `conda-forge`:

```shell
conda install -c conda-forge xdggs
```

from github:

```shell
pip install "xdggs @ git+https://github.com/xarray-contrib/xdggs.git"
```

or from source:

```shell
git clone https://github.com/xarray-contrib/xdggs.git
cd xdggs
pip install .
```

## Minimum dependency policy

```{warning}
`xdggs` is experimental and thus will switch to newer versions of dependencies if there's significant benefit.
```

Otherwise, it will follow a rolling release policy similar to {ref}`xarray:/getting-started-guide/installing.rst#minimum-dependency-versions`:

- **Python**: 30 months
- **numpy**: 12 months
- **all other libraries**: 6 months

This means that `xdggs` will be allowed to require a minor version (X.Y) once it is older than N months.
