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
copyright = "2024, Marissa Dominijanni"
author = "Marissa Dominijanni"

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
    "notfound.extension",
    "sphinx_design",
    "sphinx.ext.graphviz",
    "sphinx_remove_toctrees",
    "sphinx_toolbox.more_autosummary.column_widths",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

myst_enable_extensions = ["amsmath", "dollarmath", "smartquotes", "colon_fence"]

add_module_names = False

autosectionlabel_prefix_document = True

myst_heading_anchors = 2

graphviz_output_format = "svg"

templates_path = ["_templates"]

remove_from_toctrees = ["reference/generated/*"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"  # "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "Inferno"
# html_logo = "images/logo-anymode.svg"
html_favicon = "images/inferno-favicon.svg"

html_css_files = [
    "css/shape.css",
]

html_theme_options = {
    "light_logo": "logo-lightmode.svg",
    "dark_logo": "logo-darkmode.svg",
    "footer_icons": [
        {
            "name": "PyPi",
            "url": "https://pypi.org/project/inferno-ai/",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 17.313 19.807">
                    <path fill-rule="evenodd" d="m10.383 0.2-3.239 1.1769 3.1883 1.1614 3.239-1.1798zm-3.4152 1.2411-3.2362 1.1769 3.1855 1.1614 3.2369-1.1769zm6.7177 0.00281-3.2947 1.2009v3.8254l3.2947-1.1988zm-3.4145 1.2439-3.2926 1.1981v3.8254l0.17548-0.064132 3.1171-1.1347zm-6.6564 0.018325v3.8247l3.244 1.1805v-3.8254zm10.191 0.20931v2.3137l3.1777-1.1558zm3.2947 1.2425-3.2947 1.1988v3.8254l3.2947-1.1988zm-8.7058 0.45739c0.00929-1.931e-4 0.018327-2.977e-4 0.027485 0 0.25633 0.00851 0.4263 0.20713 0.42638 0.49826 1.953e-4 0.38532-0.29327 0.80469-0.65542 0.93662-0.36226 0.13215-0.65608-0.073306-0.65613-0.4588-6.28e-5 -0.38556 0.2938-0.80504 0.65613-0.93662 0.068422-0.024919 0.13655-0.038114 0.20156-0.039466zm5.2913 0.78369-3.2947 1.1988v3.8247l3.2947-1.1981zm-10.132 1.239-3.2362 1.1769 3.1883 1.1614 3.2362-1.1769zm6.7177 0.00213-3.2926 1.2016v3.8247l3.2926-1.2009zm-3.4124 1.2439-3.2947 1.1988v3.8254l3.2947-1.1988zm-6.6585 0.016195v3.8275l3.244 1.1805v-3.8254zm16.9 0.21143-3.2947 1.1988v3.8247l3.2947-1.1981zm-3.4145 1.2411-3.2926 1.2016v3.8247l3.2926-1.2009zm-3.4145 1.2411-3.2926 1.2016v3.8247l3.2926-1.2009zm-3.4124 1.2432-3.2947 1.1988v3.8254l3.2947-1.1988zm-6.6585 0.019027v3.8247l3.244 1.1805v-3.8254zm13.485 1.4497-3.2947 1.1988v3.8247l3.2947-1.1981zm-3.4145 1.2411-3.2926 1.2016v3.8247l3.2926-1.2009zm2.4018 0.38127c0.0093-1.83e-4 0.01833-3.16e-4 0.02749 0 0.25633 0.0085 0.4263 0.20713 0.42638 0.49826 1.97e-4 0.38532-0.29327 0.80469-0.65542 0.93662-0.36188 0.1316-0.65525-0.07375-0.65542-0.4588-1.95e-4 -0.38532 0.29328-0.80469 0.65542-0.93662 0.06842-0.02494 0.13655-0.03819 0.20156-0.03947zm-5.8142 0.86403-3.244 1.1805v1.4201l3.244 1.1805z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/mdominijanni/inferno",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# notfound_template = "404.rst"

# pygments_style = "sphinx"
# pygments_dark_style = "dracula"
