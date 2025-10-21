# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "Kubetorch"
copyright = "Runhouse Inc"
author = "the Runhouse team 🏃‍♀️🏠"

# The full version, including alpha/beta/rc tags
import kubetorch

release = kubetorch.__version__


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "_ext.json_globaltoc",
]

autodoc_typehints_format = "short"
autodoc_default_flags = ["members", "show-inheritance"]
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

markdown_http_base = "/docs/guide"
markdown_anchor_sections = True

if tags.has("json"):
    # Force simpler output format (helps CLI output)
    autodoc_typehints = "signature"  # "description"
    napoleon_use_param = True
    napoleon_use_rtype = True

    html_link_suffix = ""
    json_baseurl = "docs/"

# -- Options for HTML output -------------------------------------------------

if not tags.has("json"):
    html_theme = "sphinx_book_theme"

html_title = "Kubetorch"
html_theme_options = {
    "path_to_docs": "docs/",
    "home_page_in_toc": True,
}

# -- Disable "View Source" links and code display ----------------------------

html_show_sourcelink = False  # hides "View Source" link
html_copy_source = False  # prevents .html files from containing source code
