# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))
from pystra import __version__ as ver


# -- Project information -----------------------------------------------------

project = "PySTRA"
copyright = "2023, The PySTRA Developers"
author = "Colin Caprani, Shihab Khan, Jürgen Hackl"

# The full version, including alpha/beta/rc tags
# The short Major.Minor.Build version
_v = ver.split(".")
_build = "".join([c for c in _v[2] if c.isdigit()])
version = _v[0] + "." + _v[1] + "." + _build
release = ver


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # See https://github.com/tox-dev/sphinx-autodoc-typehints/issues/15
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    # .. "recommonmark",
    "nbsphinx",
]

autodoc_member_order = "bysource"
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
add_module_names = False  # Remove namespaces from class/method signatures
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_path = [
    "_themes",
]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pystra/pystra",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/ccaprani",
            "icon": "fab fa-twitter-square",
        },
    ],
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "pystra",
    "github_repo": "pystra",
    "github_version": "main",
    "doc_path": "docs/source/",
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "./images/logo/icon_pystra_small.png"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
