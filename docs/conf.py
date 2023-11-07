# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Inferno"
copyright = "2023, MD"
author = "MD"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.graphviz",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "attrs": ("https://www.attrs.org/en/stable/", None),
}

myst_enable_extensions = ["amsmath", "dollarmath", "smartquotes"]

add_module_names = False

autosectionlabel_prefix_document = True

myst_heading_anchors = 2

graphviz_output_format = 'svg'

templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_theme = 'pydata_sphinx_theme'
html_static_path = ["_static"]

html_title = "Inferno"
