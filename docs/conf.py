import os
import sys


sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))


project = 'TS IS FRESH'
copyright = '2024, Zinchenko Sergey'
author = 'Zinchenko Sergey'
release = '0.0.1'

pygments_style = "sphinx"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.imgmath",
    "sphinx.ext.napoleon",
    'sphinx.ext.autosectionlabel'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#e0b97e',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_static_path = ['_static']

html_context = {
    'display_github': True,
    'github_repo': 'https://github.com/zinchse/ts-is-fresh',
    'guthub_button': True
}