"""Sphinx documentation; render locally executed notebook outputs without execution."""
project = 'NeuralJAXwork'
author = 'Rudramani Singha'
copyright = '2026, Rudramani Singha'
release = '0.1.0'
extensions = ['myst_nb', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.githubpages']
html_theme = 'pydata_sphinx_theme'
html_title = 'NeuralJAXwork'
html_logo = 'logo/jax_logo_min.png'
html_favicon = 'logo/jax_logo_min.png'
html_static_path = ['docs/_static']
html_css_files = ['custom.css']
templates_path = ['docs/_templates']
html_theme_options = {
    'github_url': 'https://github.com/rgs2151/NeuralJAXwork',
    'logo': {'text': 'NeuralJAXwork'},
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'search_bar_text': 'Search documentation',
    'search_as_you_type': True,
    'show_nav_level': 2,
}
# Training and notebook execution happen locally, never during a docs build.
nb_execution_mode = 'off'
nb_merge_streams = True
exclude_patterns = ['.venv/**', '_build/**', 'legacy/**', 'design/**', 'maintainers/**',
                    'data/**', 'runs/**', 'build/**', 'dist/**', '.pytest_cache/**',
                    '.jupyter_cache/**', '**/.ipynb_checkpoints/**',
                    'README.md', 'docs/_templates/**', 'notebooks/README.md', 'examples/README.md']
suppress_warnings = ['myst.xref_missing']
