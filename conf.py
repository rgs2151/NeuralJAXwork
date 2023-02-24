# Documentation build configuration file

import os
import sys

sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'NeuralJAXwork'
copyright = '2023, Rudramani Singha'
author = 'Rudramani Singha'

# -- General configuration ---------------------------------------------------

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

html_theme = 'furo'

html_title = "NeuralJAXwork"

html_static_path = []