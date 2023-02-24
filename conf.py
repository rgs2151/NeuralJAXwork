# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NeuralJAXwork'
copyright = '2023, Rudramani Singha'
author = 'Rudramani Singha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'myst_parser',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    ]

# Add the following line to exclude classes inherited from object from the documentation
autodoc_default_options = {'members': True, 'undoc-members': True, 'private-members': False, 'special-members': "__init__", 'inherited-members': True, 'show-inheritance': True,}

# Generate the autosummary files
autosummary_generate = True
# autosummary_generate_overwrite = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'

html_title = "NeuralJAXwork"

html_static_path = []
# html_theme_options = {
#     "sidebar_hide_name": False,
#     "navigation_with_keys": False,
#     "top_of_page_button": None, #other option = Edit
# }


# # add this at the end of conf.py
# def run_apidoc(_):
#     from sphinx.ext.apidoc import main
#     parent_dir = os.path.dirname(__file__)
#     module_dir = os.path.join(parent_dir, 'NeuralJAXwork')
#     main(['-e', '-o', parent_dir, module_dir, '--force'])

# def setup(app):
#     app.connect('builder-inited', run_apidoc)
