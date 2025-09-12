# Contribute to xdggs

## Getting a development environment

Set up your local environment with either mamba or conda

::::{tab-set}
:::{tab-item} Mamba

```shell
mamba create -n xdggs-dev python=3.12
mamba env update -n xdggs-dev -f ci/environment.yml
mamba activate xdggs-dev
pip install --no-deps -e .
```

:::

:::{tab-item} Conda

```shell
conda create -n xdggs-dev python=3.12
conda env update -n xdggs-dev -f ci/environment.yml
conda activate xdggs-dev
pip install --no-deps -e .
```

:::
::::

From here you can run the tests:

```shell
pytest
```

## Running pre-commit hooks locally

In this project, we use [pre-commit](https://pre-commit.com/) to run some checks before committing code. These are run automatically in CI, but for those wanting to run them locally, install pre-commit (e.g., via [Brew](https://formulae.brew.sh/formula/pre-commit)) then:

```shell
pre-commit install
```

Now when you commit code, the pre-commit hooks will run automatically. You can also run them manually with:

```shell
pre-commit run --all-files
```

## Building the docs locally

Set up your local environment with either mamba or conda

::::{tab-set}
:::{tab-item} Mamba

```shell
mamba env create -f ci/docs.yml
mamba activate xdggs-docs
```

:::

:::{tab-item} Conda

```shell
conda env create -f ci/docs.yml
conda activate xdggs-docs
```

:::
::::

And build the documentation locally (all commands assume you are in the root repo directory)

::::{tab-set}
:::{tab-item} Automatically show changes

```
sphinx-autobuild docs docs/_build/html --open-browser
```

This will open a browser window that shows a live preview (meaning that changes you make to the configuration and content will be automatically updated and shown in the browser).
:::

:::{tab-item} Build and open manually

From the root repo diretory build the html

```shell
sphinx-build -b html docs docs/_build/html
```

and open it in a browser

```
open docs/_build/html/index.html    # macOS
xdg-open docs/_build/html/index.html  # Linux
start docs/_build/html/index.html  # Windows
```

You will have to repeat these steps when you make changes
:::
::::
