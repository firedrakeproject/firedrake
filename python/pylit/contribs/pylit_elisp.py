#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
# pylit_elisp.py: Settings and filters for elisp conversion
# ===============================================================
# 
# :Date:      $Date: 2007-05-14 14:14:54 +0200$
# :Version:   SVN-Revision $Revision$
# :URL:       $URL$
# :Copyright: 2007 Riccardo Murri
#             Released under the terms of the GNU General Public License 
#             (v. 2 or later)
# 
# .. sectnum::
# .. contents::

# Frontmatter
# ===========
# 
# Changelog
# ---------
# 
# :0.2: Complete rewrite, following G. Milde's suggestions.
# :0.3: * Rewrite filter as iterator generators, the filter interface is 
#         moved to pylit.py(GM)
#       * require three semicolons (``;;;``) in section header regexp
# ::

"""Emacs lisp for the PyLit code<->text converter.

Defaults and filters for elisp (emacs lisp) code
"""

__docformat__ = 'restructuredtext'

_version = "0.3"


# Requirements
# ------------
# 
# Regular espressions are used in `ELISP filters` to match
# the ELisp sectioning comments::

import re

import pylit

# Emacs Lisp filters
# ==================
# 
# 
# ::

def elisp_code_preprocessor(data):
    """Convert Emacs Lisp comment sectioning markers to reST comments.

    The special headers listed at:

      http://www.gnu.org/software/emacs/elisp/html_node/Library-Headers.html#Library-Headers

    are converted to reST comment lines and prefixed with the comment
    line marker (taken from keyword arguments); all other content is
    passed through unchanged.

    For instance, ELisp code:

      ;;; Code:

      ;; here begins the real coding
      (defun my-elisp-function () ...)

    is translated to:

      ;; .. |elisp> ;;; Code:

      ;; here begins the real coding
      (defun my-elisp-function () ...)
    """

# Regular expression matching the special ELisp headers::

    SECTION_PATTERN = \
        re.compile(';;; *(Change *Log|Code|Commentary|Documentation|History):',\
                       re.IGNORECASE)

# Prepend ``.. |elisp> `` to matching headers, one line at a time::
    
    for line in data:
        if SECTION_PATTERN.match(line):
            yield pylit.defaults.comment_strings["elisp"] + '.. |elisp> ' + line
        else:
            yield line


def elisp_code_postprocessor(data):
    """Convert specially-marked reST comments to Emacs Lisp code.

    In all lines, the prefix ``.. |elisp> `` (note
    the trailing space!) is stripped from the begin of a line.  

    For instance, the block:

      .. |elisp> ;;; Code:
      .. |elisp> (some-elisp-incantation)
      (another-one)

    is translated to:

      ;;; Code:
      (some-elisp-incantation)
      (another-one)
    """
    
    # Set the prefix to be stripped
    prefix = pylit.defaults.comment_strings["elisp"] + '.. |elisp> '
    
    for line in data:
        if line.startswith(prefix):
            yield line[len(prefix):]
        else:
            yield line



# Register elisp
# ==============
# 
# Add default values for "elisp" to the `defaults` object from PyLit.
# 
# The following code assumes that this plug-in is always evaluated in the
# pylit namespace and after pylit.py
# 
# ::

pylit.defaults.languages[".el"] = "elisp"
pylit.defaults.comment_strings["elisp"] = ';; '

# Filters
pylit.defaults.preprocessors["elisp2text"] = elisp_code_preprocessor
pylit.defaults.postprocessors["text2elisp"] = elisp_code_postprocessor

