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
import hashlib
from importlib import metadata
import os
from pathlib import Path
import sys

src_path = str(Path(__file__).resolve().parents[2] / "src")
sys.path.insert(0, src_path)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [src_path, os.environ["PYTHONPATH"]] if "PYTHONPATH" in os.environ else [src_path]
)
from pystra import __version__ as ver

# -- Project information -----------------------------------------------------

project = "PySTRA"
copyright = "2021-2026, The PySTRA Developers"
author = "Colin Caprani, Shihab Khan, Jürgen Hackl"

# Retain the development suffix in every documentation version label.
version = ver
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
    "myst_parser",
    "sphinx_copybutton",
]

autodoc_member_order = "bysource"
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
# Execute notebooks when Sphinx reads them. Its document cache reuses unchanged
# notebooks; the hooks below also invalidate them when their Python inputs change.
nbsphinx_execute = "always"
nbsphinx_kernel_name = "python3"
nbsphinx_allow_errors = False
add_module_names = False  # Remove namespaces from class/method signatures
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {".rst": "restructuredtext", ".md": "restructuredtext"}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    # Legacy files superseded by current notebooks/content
    "notebooks/intro.rst",
    "notebooks/ex_code_calibration.ipynb",
    "notebooks/ex_openseespy.ipynb",
    "notebooks/example_global_calibration.ipynb",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_path = [
    "_themes",
]

html_theme_options = {
    "header_links_before_dropdown": 5,
    "navbar_start": ["navbar-logo", "version-status"],
    "footer_end": ["project-links"],
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
    "github_version": "v2.0",
    "doc_path": "docs/source/",
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "./images/logo/icon_pystra_small.png"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# Copy executable inputs, not notebook outputs or console prompts.
copybutton_selector = "div.highlight pre:not(.nboutput pre)"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


def _execution_signature():
    """Conservatively track code and environment used by notebook kernels."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = sorted((root / "src/pystra").rglob("*.py"))
    paths += sorted((root / "docs/source/notebooks").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    versions = sorted(
        (distribution.metadata.get("Name", ""), distribution.version)
        for distribution in metadata.distributions()
    )
    digest.update(repr((sys.executable, sys.version, versions)).encode())
    return digest.hexdigest()


def _refresh_notebooks(app, env, added, changed, removed):
    if app.config.nbsphinx_execute != "always":
        return []
    signature = _execution_signature()
    app._pystra_execution_signature = signature
    if getattr(env, "pystra_execution_signature", None) == signature:
        return []
    return [
        name
        for name in env.found_docs - set(removed)
        if (Path(app.srcdir) / f"{name}.ipynb").is_file()
    ]


def _record_execution_signature(app, env):
    if app.config.nbsphinx_execute == "always":
        env.pystra_execution_signature = app._pystra_execution_signature


def setup(app):
    app.connect("env-get-outdated", _refresh_notebooks)
    app.connect("env-updated", _record_execution_signature)


# Links shared by generated module and class pages.
autosummary_context = {
    "pystra_api_routes": {
        "model": ("guides/models", "ex_first_analysis", "fundamentals"),
        "analysis": ("guides/models", "ex_first_analysis", "fundamentals"),
        "distributions": ("copulas", "ex_copulas", "transformations"),
        "copula": ("copulas", "ex_copulas", "transformations"),
        "joint": ("copulas", "ex_copulas", "transformations"),
        "transformation": ("copulas", "ex_copulas", "transformations"),
        "correlation": ("copulas", "ex_copulas", "transformations"),
        "integration": ("copulas", "ex_copulas", "transformations"),
        "quadrature": ("copulas", "ex_copulas", "transformations"),
        "calibration": (
            "guides/calibration",
            "ex_generic_calibration",
            "code_calibration",
        ),
        "loadcomb": (
            "guides/calibration",
            "ex_generic_calibration",
            "code_calibration",
        ),
        "fbc": ("guides/calibration", "ex_generic_calibration", "code_calibration"),
        "ddo": ("guides/assessment", "ex_design_decision_optimization", "decisions"),
        "active_learning": ("active_learning", "ex_active_learning", "active_learning"),
        "plotting": ("plotting", "ex_generic_calibration", "code_calibration"),
        "form": ("guides/methods", "ex_intro", "design_point_methods"),
        "results": ("guides/methods", "ex_intro", "design_point_methods"),
        "sorm": ("guides/methods", "ex_intro", "design_point_methods"),
        "mc": ("guides/methods", "ex_intro", "design_point_methods"),
        "ls": ("guides/methods", "ex_intro", "design_point_methods"),
        "ss": ("guides/methods", "ex_intro", "design_point_methods"),
        "sensitivity": ("guides/methods", "ex_intro", "design_point_methods"),
        "system": ("guides/methods", "ex_intro", "design_point_methods"),
        "system_form": ("guides/methods", "ex_intro", "design_point_methods"),
        "strong_maximum": ("guides/methods", "ex_intro", "design_point_methods"),
    }
}

sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))
extensions.append("pystra_docs")
exclude_patterns.append("_generated/**")
nbsphinx_prolog = r"""
{% set notebook = env.docname.split('/')[-1] ~ '.ipynb' %}
{% set details = env.pystra_notebooks[env.docname] %}
.. container:: notebook-actions

   :download:`Download notebook <{{ notebook }}>` · :download:`Download runnable bundle <../_generated/notebooks/{{ notebook[:-6] }}.zip>`

   {% if details.dependencies == 'al' %}**Dependencies:** PySTRA with the optional ``al`` extra.{% else %}**Dependencies:** PySTRA core.{% endif %}
   {% if details.support_files %}**Helper files:** {{ details.support_files | join(', ') }} (included in the bundle).{% endif %}
"""
