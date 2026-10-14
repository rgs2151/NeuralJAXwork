"""Executable Sphinx tutorial book and API reference."""
project = 'NeuralJAXwork'
author = 'Rudramani Singha'
copyright = '2026, Rudramani Singha'
release = '0.1.0'
extensions = ['myst_nb', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.githubpages']
html_theme = 'sphinx_book_theme'
html_title = 'NeuralJAXwork — models, training, and tutorials'
html_theme_options = {
    'repository_url': 'https://github.com/rgs2151/NeuralJAXwork',
    'repository_branch': 'main',
    'use_repository_button': True,
    'use_download_button': True,
    'show_navbar_depth': 2,
}
# Always execute the tiny offline notebook configuration. A failing cell fails CI.
nb_execution_mode = 'force'
nb_execution_timeout = 90
nb_execution_raise_on_error = True
nb_execution_allow_errors = False
nb_execution_in_temp = False
nb_merge_streams = True
exclude_patterns = ['.venv/**', '_build/**', 'legacy/**', 'design/**',
                    'data/**', 'runs/**', 'build/**', 'dist/**', '.pytest_cache/**',
                    '.jupyter_cache/**', '**/.ipynb_checkpoints/**',
                    'README.md', 'notebooks/README.md', 'examples/README.md']
suppress_warnings = ['myst.xref_missing']
