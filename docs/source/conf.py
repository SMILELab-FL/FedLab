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

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, target_dir)
# print(target_dir)

# -- Project information -----------------------------------------------------

project = 'FedLab'
copyright = '2021, SMILE Lab'
author = 'SMILE Lab'

# The full version, including alpha/beta/rc tags
import fedlab

release = fedlab.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',  # this one is really important
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinxcontrib.napoleon',
    'sphinx.ext.autosectionlabel',  # allows referring sections its title, affects `ref`
    'sphinx_design',
    'sphinxcontrib.bibtex',
]

# for 'sphinxcontrib.bibtex' extension
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrt'

autodoc_mock_imports = ["numpy", "torch", "torchvision", "pandas"]
autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# configuration for 'autoapi.extension'
autoapi_type = 'python'
autoapi_dirs = ['../../fedlab']
autoapi_template_dir = '_autoapi_templates'
add_module_names = False  # makes Sphinx render package.module.Class as Class

# Add more mapping for 'sphinx.ext.intersphinx'
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'PyTorch': ('http://pytorch.org/docs/master/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/dev/', None)}

# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Config for 'sphinx.ext.todo'
todo_include_todos = True

# multi-language docs
language = 'en'
locale_dirs = ['../locales/']   # path is example but recommended.
gettext_compact = False  # optional.
gettext_uuid = True  # optional.

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "furo"
html_favicon = "../imgs/favicon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../_static']
html_theme_options = {
    # "announcement": """
    #     <a style=\"text-decoration: none; color: white;\"
    #        href=\"https://github.com/sponsors/urllib3\">
    #        <img src=\"/en/latest/_static/favicon.png\"/> Support urllib3 on GitHub Sponsors
    #     </a>
    # """,
    "sidebar_hide_name": True,
    "light_logo": "FedLab-logo.svg",
    "dark_logo": "FedLab-logo.svg",
}
# html_logo = "FedLab-logo.svg"
