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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- Project information -----------------------------------------------------

project = 'PyAnomaly'
copyright = '2024, Chulwoo Han and Jongho Kang'
author = 'Chulwoo Han and Jongho Kang'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    #    'sphinxcontrib.bibtex'
 ]

autodoc_default_options = {
    # 'members': 'var1, var2',
    'member-order': 'alphabetical',  # 'bysource', 'alphabetical'
    # 'special-members': '__init__',
    # 'undoc-members': True,
    # 'exclude-members': '__weakref__'
}

# generate autosummary even if no references
# autosummary_generate = True
# autosummary_imported_members = True

# pygments_style = 'sphinx'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 2,  # Depth of the menu on the left panel.
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

todo_include_todos = False

autodoc_mock_imports = ['numpy', 'pandas', 'wrds', 'numba', 'scipy', 'statsmodels', 'openpyxl', 'matplotlib', 'sklearn']

# -- rst prolog
# Substitutions
rst_prolog = """
.. |email| replace:: chulwoo.han@skku.edu
.. _PyAnomaly repository: https://github.com/chulwoohan/pyanomaly
.. _mapping file: https://github.com/chulwoohan/pyanomaly/blob/master/mapping.xlsx
.. _examples: https://github.com/chulwoohan/pyanomaly
.. _CZ's openassetpricing: https://www.openassetpricing.com/
.. _GHZ' SAS code: https://sites.google.com/site/jeremiahrgreenacctg/home
.. _JKP's SAS code: https://github.com/bkelly-lab/ReplicationCrisis

"""

