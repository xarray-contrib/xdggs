# -- Project information -----------------------------------------------------
import datetime as dt

import sphinx_autosummary_accessors

import xdggs  # noqa: F401

project = "xdggs"
author = f"{project} developers"
initial_year = "2023"
year = dt.datetime.now().year
copyright = f"{initial_year}-{year}, {author}"

# The root toctree document.
root_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",  # enables myst-nb support for plain md files
}

extensions = [
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_autosummary_accessors",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

extlinks = {
    "issue": ("https://github.com/xarray-contrib/xdggs/issues/%s", "GH%s"),
    "pull": ("https://github.com/xarray-contrib/xdggs/pull/%s", "PR%s"),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "directory"]

## Github Buttons
html_theme_options = {
    "repository_url": "https://github.com/xarray-contrib/xdggs",
    "use_repository_button": True,
    "use_issues_button": True,
}

# -- autosummary / autodoc ---------------------------------------------------

autosummary_generate = True
autodoc_typehints = "none"

# -- napoleon ----------------------------------------------------------------

napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict_like": ":term:`dict-like <mapping>`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    "file-like": ":term:`file-like <file-like object>`",
    "any": ":py:class:`any <object>`",
    # numpy terms
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
    "scalar": ":term:`scalar`",
    "array": ":term:`array`",
    "hashable": ":term:`hashable <name>`",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = "_static/logos/xdggs_logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# -- Options for the intersphinx extension -----------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "lonboard": ("https://developmentseed.org/lonboard/latest", None),
    "healpy": ("https://healpy.readthedocs.io/en/latest", None),
    "cdshealpix-python": ("https://cds-astro.github.io/cds-healpix-python", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
}

# -- myst-nb options ---------------------------------------------------------

nb_execution_timeout = -1
nb_execution_cache_path = "_build/myst-nb"

# myst options ---------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",  # Enables ::: directive syntax
    "deflist",
    "html_admonition",
    "html_image",
    "replacements",
    "substitution",
]

# -- sphinxcontrib-bibtex ----------------------------------------------------

bibtex_bibfiles = ["reference_guide/publications.bib"]
