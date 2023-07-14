#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# Test the pylit.py literal python module
# =======================================
# 
# :Date:      $Date$
# :Version:   SVN-Revision $Revision$
# :URL:       $URL: svn+ssh://svn.berlios.de/svnroot/repos/pylit/trunk/test/pylit_test.py $
# :Copyright: 2006 Guenter Milde.
#             Released under the terms of the GNU General Public License
#             (v. 2 or later)
# 
# .. contents::

from pprint import pprint
# import operator
from pylit import *
from pylit_elisp import *
import nose

# Test source samples
# ===================
#

code = {}
filtered_code = {}
text = {}

code["simple"] = [";; documentation::\n",
                  "\n",
                  "code_block\n"]
filtered_code["simple"] = code["simple"]

text["simple"] = ["documentation::\n",
                  "\n",
                  "  code_block\n"]


code["section header"] = [";; \n", ";;;Commentary:\n"]
filtered_code["section header"] = [";; \n", ";; .. |elisp> ;;;Commentary:\n"]
text["section header"] = ["\n", ".. |elisp> ;;;Commentary:\n"]


# This example fails, as the rst-comment in the first text line is recognized
# as a leading code_block (header).

# code["section header"] = [";;;Commentary:\n"]
# filtered_code["section header"] = [";; .. |elisp> ;;;Commentary:\n"]
# text["section header"] = [".. |elisp> ;;;Commentary:\n"]



code["section"] = [";; \n",
                   ";;;Commentary:\n",
                   ";; This is\n",
                   ";; a test."]
filtered_code["section"] = [";; \n",
                            ";; .. |elisp> ;;;Commentary:\n",
                            ";; This is\n",
                            ";; a test."]
text["section"] = ["\n",
                   ".. |elisp> ;;;Commentary:\n",
                   "This is\n",
                   "a test."]


def test_elisp_code_preprocessor():
    """test the code preprocessing filter"""
    for key in code.keys():
        data = code[key]
        soll = filtered_code[key]
        output = [line for line in elisp_code_preprocessor(data)]
        print "ist  %r (%s)"%(output, key)
        print "soll %r (%s)"%(soll, key)
        assert output == soll
    

def test_elisp_code_postprocessor():
    """test the code preprocessing filter"""
    for key in code.keys():
        data = filtered_code[key]
        soll = code[key]
        output = [line for line in elisp_code_postprocessor(data)]
        print "ist  %r (%s)"%(output, key)
        print "soll %r (%s)"%(soll, key)
        assert output == soll
    
def test_elisp_settings():
    assert defaults.languages[".el"] == "elisp"
    assert defaults.comment_strings["elisp"] == ';; '
    assert defaults.preprocessors["elisp2text"] == elisp_code_preprocessor
    assert defaults.postprocessors["text2elisp"] == elisp_code_postprocessor

def test_elisp2text():
    for key in code.keys():
        data = code[key]
        soll = text[key]
        converter = Code2Text(data, language="elisp")
        output = converter()
        print "ist  %r (%s)"%(output, key)
        print "soll %r (%s)"%(soll, key)
        assert output == soll

class test_Code2Text(object):
    def test_setup(self):
        converter = Code2Text(text['simple'], language="elisp")
        assert converter.preprocessor == elisp_code_preprocessor

class test_Text2Code(object):
    def test_setup(self):
        converter = Text2Code(text['simple'], language="elisp")
        assert converter.postprocessor == elisp_code_postprocessor

    def test_call_without_filter(self):
        for key in code.keys():
            data = text[key]
            soll = filtered_code[key]
            converter = Text2Code(data, comment_string=";; ")
            output = converter()
            print "ist  %r (%s)"%(output, key)
            print "soll %r (%s)"%(soll, key)
            assert output == soll

    def test_convert(self):
        for key in code.keys():
            data = text[key]
            soll = filtered_code[key]
            converter = Text2Code(data, language="elisp")
            output = [line for line in converter.convert(data)]
            print "ist  %r (%s)"%(output, key)
            print "soll %r (%s)"%(soll, key)
            assert output == soll

    def test_call_with_filter(self):
        for key in code.keys():
            data = text[key]
            soll = code[key]
            converter = Text2Code(data, language="elisp")
            output = converter()
            print "ist  %r (%s)"%(output, key)
            print "soll %r (%s)"%(soll, key)
            assert output == soll


if __name__ == "__main__":
    nose.runmodule() # requires nose 0.9.1
    sys.exit()
