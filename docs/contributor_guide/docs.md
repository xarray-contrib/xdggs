# Contribute to the Documentation

## Building the docs locally

Set up your local environment with either mamba or conda

::::{tab-set}
:::{tab-item} Mamba

```shell
mamba env create -f ci/docs.yml
```

:::

:::{tab-item} Conda

```shell
conda env create -f ci/docs.yml
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
