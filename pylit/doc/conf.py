## -*- coding: utf-8 -*-
  
# conf.py -- Sphinx configuration for PyLit
# =========================================
# 
# Documentation build configuration file
# 
# This file is execfile()d with the current directory set to its containing
# dir.
# 
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed
# automatically).
# 
# All configuration values have a default value; values that are commented out
# serve to show the default value. 
# 
# You can use the following modules in your definitions (or add more)::

import sys, os

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use ``os.path.abspath`` to make it
# absolute, like shown here::

#sys.path.append(os.path.abspath('some/directory'))

# General configuration
# ---------------------
# 
# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones. ::

extensions = []

# Add any paths that contain templates here, relative to this directory::

templates_path = ['.templates']

# The suffix of source filenames::

source_suffix = '.txt'

# The master toctree document::

master_doc = 'index'

# General substitutions::

project = 'PyLit'
copyright = u'2009, Günter Milde'

# The default replacements for ``|version|`` and ``|release|``, also used in
# various other places throughout the built documents.
# ::

# The short X.Y version.
version = '0.7'
# The full version, including alpha/beta/rc tags.
release = '0.7.9'

# There are two options for replacing ``|today|``: either, you set today to
# some non-false value, then it is used::

#today = ''

# Else, today_fmt is used as the format for a strftime call::

today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build::

#unused_docs = ["rstdocs/download/index.txt", "tutorial/*.py.txt"]

# A list of glob-style patterns that should be excluded when looking for
# source files. [1] They are matched against the source file names relative
# to the source directory, using slashes as directory separators on all
# platforms.

exclude_patterns = ['**/.svn']

# Deprecated since version 1.0: Use exclude_patterns instead.
# exclude_dirnames = [".svn"]

# The name of the default domain. Can also be None to disable a default
# domain. The default is 'py'. Those objects in other domains (whether the
# domain name is given explicitly, or selected by a default-domain directive)
# require the domain name explicitly prepended when named.

# primary_domain = 'py'
primary_domain = None


# The reST default role (used for this markup: ```text```) to use for all
# documents::

#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text. ::

add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::) ::

#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default. ::

#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
# Can be any registered pygments-style or 'sphinx'.
# 
# >>> from pygments.styles import STYLE_MAP
# >>> STYLE_MAP.keys()
# ['manni', 'perldoc', 'borland', 'colorful', 'default', 'murphy', 'trac', 
#  'fruity', 'autumn', 'bw', 'emacs', 'pastie', 'friendly', 'native']
# 
# You can try the styles with the  `pygments demo
# <http://pygments.org/demo/>`_ pages that offer a drop-down list to select
# the style, e.g. http://pygments.org/demo/1444/
# 
# 
# ::

#pygments_style = 'sphinx' 
pygments_style = 'friendly'

# Options for HTML output
# -----------------------
# 
# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path::

html_style = 'pylit-sphinx.css'
# html_style = 'sphinxdoc.css'

# Options to the theme, like a sidebar that is visible even when
# scrolling (TODO: how to get this to work (maybe just update Spinx)?)::

#html_theme_options = {'stickysidebar': 'true'}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation". ::

html_title = "PyLit"

# A shorter title for the navigation bar.  Default is the same as html_title.
# ::

html_short_title = "Home"

# The name of an image file (within the static path) to place at the top of
# the sidebar. 
# Korrektion [GM]: path is relative to source (not static) 
# (Bug or config issue?). 
# ::

html_logo = "logo/pylit-bold-framed.png"
# html_logo = "pylit-bold-framed.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large. ::

html_favicon = "favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# ::

html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format::

html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities::

#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names::

#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names::

#html_additional_pages = {}

# If false, no module index is generated::

html_use_modindex = False

# If false, no index is generated::

html_use_index = False

# If true, the index is split into individual pages for each letter::

#html_split_index = False

# If true, the reST sources are included in the HTML build as _sources/<name>.
# (needed for the search feature)::

#html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served. ::

#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files
# (e.g. ".xhtml" for correct MathML rendering in Firefox)::

#html_file_suffix = '' 

# Output file base name for HTML help builder::

htmlhelp_basename = 'PyLit-doc'

# Delimiter in the relbar: Fallback: ' &raquo;' ::

reldelim1 = ' / '

# Options for LaTeX output
# ------------------------
# 
# The paper size ('letter' or 'a4')::

#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt')::

#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples (source start
# file, target name, title, author, document class [howto/manual])::

latex_documents = [
  ('index', 'PyLit.tex', 'PyLit Documentation',
   u'Günter Milde', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top
# of the title page::

#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters::

#latex_use_parts = False

# Additional stuff for the LaTeX preamble::

#latex_preamble = ''

# Documents to append as an appendix to all manuals::

#latex_appendices = []

# If false, no module index is generated::

#latex_use_modindex = True
