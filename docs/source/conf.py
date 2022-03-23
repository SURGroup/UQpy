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

sys.path.insert(0, os.path.abspath("./src/UQpy/docs"))

# -- Project information -----------------------------------------------------

project = "UQpy"
copyright = "2021, Michael D. Shields"
author = (
    "Michael D. Shields, Dimitrs G. Giovanis, Audrey Olivier, B.S. Aakash, Mohit Singh Chauhan, Lohit Vandanapu, "
    "Ketson RM dos Santos"
)

# The full version, including alpha/beta/rc tags
release = "v4.0.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "nbsphinx",
    "sphinx_gallery.load_style",
]

nbsphinx_custom_formats={
    ".md": ["jupytext.reads", {"fmt": "mystnb"}]
}
autoclass_content = 'init'
add_module_names = False
autodoc_member_order = 'bysource'
napoleon_use_param = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'unsrt'

# Try to remove duplicate labels
autosectionlabel_prefix_document = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "Model_Runs**"]

sphinx_gallery_conf = {
    'examples_dirs': ['../code/dimension_reduction/diffusion_maps',
                      '../code/dimension_reduction/pod',
                      '../code/dimension_reduction/grassmann',
                      '../code/distributions/continuous_1d',
                      '../code/distributions/discrete_1d',
                      '../code/distributions/multivariate',
                      '../code/distributions/user_defined',
                      '../code/sampling/adaptive_kriging',
                      '../code/sampling/importance_sampling',
                      '../code/sampling/monte_carlo',
                      '../code/sampling/latin_hypercube',
                      '../code/sampling/mcmc',
                      '../code/sampling/simplex',
                      '../code/sampling/true_stratified_sampling',
                      '../code/sampling/refined_stratified_sampling',
                      '../code/inference/mle',
                      '../code/inference/info_model_selection',
                      '../code/inference/bayes_parameter_estimation',
                      '../code/inference/bayes_model_selection',
                      '../code/transformations/nataf',
                      '../code/sensitivity/morris',
                      '../code/stochastic_processes/bispectral',
                      '../code/stochastic_processes/karhunen_loeve',
                      '../code/stochastic_processes/spectral',
                      '../code/stochastic_processes/translation',
                      '../code/reliability/form',
                      '../code/reliability/sorm',
                      '../code/reliability/subset_simulation',
                      '../code/surrogates/srom',
                      '../code/surrogates/gpr',
                      '../code/surrogates/pce',
                      '../code/RunModel',],  # path to your example scripts,
    'gallery_dirs': ['auto_examples/dimension_reduction/diffusion_maps',
                     'auto_examples/dimension_reduction/pod',
                     'auto_examples/dimension_reduction/grassmann',
                     'auto_examples/distributions/continuous_1d',
                     'auto_examples/distributions/discrete_1d',
                     'auto_examples/distributions/multivariate',
                     'auto_examples/distributions/user_defined',
                     'auto_examples/sampling/adaptive_kriging',
                     'auto_examples/sampling/importance_sampling',
                     'auto_examples/sampling/monte_carlo',
                     'auto_examples/sampling/latin_hypercube',
                     'auto_examples/sampling/mcmc',
                     'auto_examples/sampling/simplex',
                     'auto_examples/sampling/true_stratified_sampling',
                     'auto_examples/sampling/refined_stratified_sampling',
                     'auto_examples/inference/mle',
                     'auto_examples/inference/info_model_selection',
                     'auto_examples/inference/bayes_parameter_estimation',
                     'auto_examples/inference/bayes_model_selection',
                     'auto_examples/transformations/nataf',
                     'auto_examples/sensitivity/morris',
                     'auto_examples/stochastic_processes/bispectral',
                     'auto_examples/stochastic_processes/karhunen_loeve',
                     'auto_examples/stochastic_processes/spectral',
                     'auto_examples/stochastic_processes/translation',
                     'auto_examples/reliability/taylor',
                     'auto_examples/reliability/subset_simulation',
                     'auto_examples/surrogates/srom',
                     'auto_examples/surrogates/gpr',
                     'auto_examples/surrogates/pce',
                     'auto_examples/RunModel',],  # path to where to save gallery generated output
    'binder': {
        # Required keys
        'org': 'SURGroup',
        'repo': 'UQpy',
        'branch': 'master',  # Can be any branch, tag, or commit hash. Use a branch that hosts your docs.
        'binderhub_url': 'https://mybinder.org',
        # Any URL of a binderhub deployment. Must be full URL (e.g. https://mybinder.org).
        'dependencies': './binder/requirements.txt',
        'notebooks_dir': 'notebooks',
        'use_jupyter_lab': True
        # Jupyter notebooks for Binder will be copied to this directory (relative to built documentation root).
    },
    'ignore_pattern': '/local_',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
language = None
pygments_style = None

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'logo_only': True,
    'style_nav_header_background': '#F0F0F0',
    'vcs_pageview_mode': 'view'
}

github_url = "https://github.com/SURGroup/UQpy"
html_logo = "_static/logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_sidebars = {
    "**": ["about.html", "navigation.html", "relations.html", "searchbox.html", ]
}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']
# source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "uqpydoc"

# autodoc_default_flags = ['members',  'private-members']

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "uqpy.tex", "UQpy Documentation", "Michael D. Shields", "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "uqpy", "UQpy Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "uqpy",
        "UQpy Documentation",
        author,
        "uqpy",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
