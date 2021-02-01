#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

## pylit_test.py
## *************
## Test pylit.py Python Module
## +++++++++++++++++++++++++++
##
## :Date:      $Date: 2007-05-17 $
## :Version:   SVN-Revision $Revision: 45 $
## :URL:       $URL: svn+ssh://svn.berlios.de/svnroot/repos/pylit/trunk/test/pylit_test.py $
## :Copyright: 2006 Guenter Milde.
##             Released under the terms of the GNU General Public License
##             (v. 2 or later)
##
## .. contents::
##
## A catalogue of errors
## =====================
##
## from file:///home/milde/Texte/Doc/Programmierung/Software-Carpentry/lec/unit.html
##
## * Numbers: zero, largest, smallest magnitude, most negative
## * Structures: empty, exactly one element, maximum number of elements
##   - Duplicate elements (e.g., letter "J" appears three times in a string)
##   - Aliased elements (e.g., a list contains two references to another list)
##   - Circular structures (e.g., a list that contains a reference to itself)
## * Searching: no match found, one match found, multiple matches found,
##   everything matches
##   - Code like x = find_all(structure)[0] is almost always wrong
##   - Should also check aliased matches (same thing found multiple times)
## ::

"""pylit_test.py: test the "literal python" module"""

from pprint import pprint
import operator
from pylit import *
import nose

## Test DefaultDict
## ================

class test_DefaultDict(object):
    """Test the DefaultDict dictionary with custom default"""
    def setUp(self):
        self.defdict = DefaultDict('#')

    def test_get_default(self):
        assert self.defdict['nonexisting'] == '#'

    def test_set_get(self):
        self.defdict['mykey'] = 3
        assert self.defdict['mykey'] == 3

    def test_change_default(self):
        self.defdict.default = '%'
        assert self.defdict['nonexisting'] == '%'

    def test_init_args(self):
        di = DefaultDict('#', {'mykey': 3})
        assert di['mykey'] == 3
        assert di['nonexisting'] == '#'

    def test_init_args2(self):
        di = DefaultDict('#', mykey = 3)
        assert di['mykey'] == 3
        assert di['nonexisting'] == '#'

    def test_init_args3(self):
        di = DefaultDict('#', [('mykey', 3)])
        assert di['mykey'] == 3
        assert di['nonexisting'] == '#'


## Text <-> Code conversion
## ========================
##
## Test strings
## ============
##
## Example of text, code and stripped code with typical features"::

text = """..  #!/usr/bin/env python
  # -*- coding: iso-8859-1 -*-
  
Leading text

in several paragraphs followed by a literal block::

  block1 = 'first block'
  
Some more text and the next block. ::

  block2 = 'second block'
  print block1, block2
  
Trailing text.
"""
# print text

## The converter expects the data in separate lines (iterator or list)
## with trailing newlines. We use the `splitlines` string method with
## `keepends=True`::

textdata = text.splitlines(True)
# print textdata

## If a "code" source is converted with the `strip` option, only text blocks
## are extracted, which leads to::

stripped_text = """Leading text

in several paragraphs followed by a literal block:

Some more text and the next block.

Trailing text.
"""

## The code corresponding to the text test string.
##
## Using a triple-quoted string for the code (and stripped_code) can create
## problems with the conversion of this test by pylit (as the text parts
## would be converted to text).
## A workaround is using a different comment string for the text blocks and
## converting with e.g. ``pylit --comment-string='## ' pylit_test.py``.
##
## ::

code = """#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# Leading text
# 
# in several paragraphs followed by a literal block::

block1 = 'first block'

# Some more text and the next block. ::

block2 = 'second block'
print block1, block2

# Trailing text.
"""
# print code

codedata = code.splitlines(True)

## Converting the text teststring with the `strip` option leads to::

stripped_code = """#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

block1 = 'first block'

block2 = 'second block'
print block1, block2

"""

## pprint(textdata)
## pprint(stripped_code.splitlines(True))
##
## Containers for special case examples:
##
## 1. Text2Code samples
## ``textsamples["what"] = (<text data>, <output>, <output (with `strip`)``
## ::

textsamples = {}

## 2. Code2Text samples
## ``codesamples["what"] = (<code data>, <output>, <output (with `strip`)``
## ::

codesamples = {}

## Auxiliary function to test the textsamples and codesamples::

def check_converter(key, converter, output):
    print "E:", key
    extract = converter()
    print(extract)
    outstr = "".join(extract)
    print "soll:", repr(output)
    print "ist: ", repr(outstr)
    assert output == outstr

## Test generator for textsample tests::

def test_Text2Code_samples():
    for key, sample in textsamples.iteritems():
        yield (check_converter, key,
               Text2Code(sample[0].splitlines(True)), sample[1])
        if len(sample) == 3:
            yield (check_converter, key,
                   Text2Code(sample[0].splitlines(True), strip=True),
                   sample[2])

## Test generator for codesample tests::

def test_Code2Text_samples():
    for key, sample in codesamples.iteritems():
        yield (check_converter, key,
               Code2Text(sample[0].splitlines(True)), sample[1])
        if len(sample) == 3:
            yield (check_converter, key,
                   Code2Text(sample[0].splitlines(True), strip=True),
                   sample[2])

## Pre and postprocessing filters (for testing the filter hooks)
##
## ::

def r2l_filter(data):
    print "applying r2l filter"
    for line in data:
        yield line.replace("r", "l")

## ::

defaults.preprocessors["rl2text"] = r2l_filter

## ::

def l2r_filter(data):
    print "applying l2r filter"
    for line in data:
        yield line.replace("l", "r")

## ::

defaults.preprocessors["text2rl"] = l2r_filter

## ::

def x2u_filter(data):
    print "applying x2u filter"
    for line in data:
        yield line.replace("x", "u")

## ::

defaults.postprocessors["x2text"] = x2u_filter

## ::

def u2x_filter(data):
    print "applying u2x filter"
    for line in data:
        yield line.replace("u", "x")

## ::

defaults.postprocessors["text2x"] = u2x_filter

## ::

def test_x2u_filter():
    soll = text.replace("x", "u")
    result = "".join([line for line in x2u_filter(textdata)])
    print "soll", repr(text)
    print "ist", repr(result)
    assert soll == result



## TextCodeConverter
## =================
##
## ::

class test_TextCodeConverter(object):
    """Test the TextCodeConverter parent class
    """

## ::

    def check_marker_regexp_true(self, sample, converter):
        match = converter.marker_regexp.search(sample)
        print 'marker: %r; sample %r' %(converter.code_block_marker, sample)
        print 'match %r'%match
        assert match is not None

## ::

    def check_marker_regexp_false(self, sample, converter):
        print 'marker: %r; sample %r' %(converter.code_block_marker, sample)
        assert converter.marker_regexp.search(sample) is None

## ::

    def test_marker_regexp(self):
        # Samples
        literal = ['::',
                   '  ::',
                   't ::',
                   'text::',
                   ' indented::',
                   ' indented ::',
                   'more text :: ',
                   ' indented text :: ',
                   '. no-directive::',
                   'a .. directive:: somewhere::']
        directives = ['.. code-block:: python3',
                     '  .. code-block:: python3',
                     '.. code-block:: python3 listings',
                     '  .. code-block:: python3 listings']
        misses = ['.. comment string ::',
                  '.. ::',
                  'text:']
        # default code_block_marker ('::')
        self.converter = TextCodeConverter(textdata)
        assert self.converter.code_block_marker == '::'
        # self.converter is not seen by the check_marker_regexp_true() method
        for sample in literal:
            yield (self.check_marker_regexp_true, sample, self.converter)
        for sample in directives+misses:
            yield (self.check_marker_regexp_false, sample, self.converter)
        # code-block directive as marker
        self.converter = TextCodeConverter(textdata,
                                           code_block_marker='.. code-block::')
        assert self.converter.code_block_marker == '.. code-block::'
        for sample in directives:
            yield (self.check_marker_regexp_true, sample, self.converter)
        for sample in literal+misses:
            yield (self.check_marker_regexp_false, sample, self.converter)

## ::

    def test_get_indent(self):
        converter = TextCodeConverter(textdata)
        assert converter.get_indent("foo") == 0
        assert converter.get_indent(" foo") == 1
        assert converter.get_indent("  foo") == 2

## ::

    def test_collect_blocks(self):
        converter = TextCodeConverter(textdata)
        textblocks = [block for block in collect_blocks(textdata)]
        print textblocks
        assert len(textblocks) == 7, "text sample has 7 blocks"
        assert reduce(operator.__add__, textblocks) == textdata

## Text2Code
## =========
##
## ::

class test_Text2Code(object):
    """Test the Text2Code class converting rst->code"""

## ::

    def setUp(self):
        self.converter = Text2Code(textdata)

## test helper funs ::

    def test_set_state_empty(self):
        try:
            self.converter.set_state([])
            raise AssertionError, "should raise StopIteration"
        except StopIteration:
            pass

    def test_set_state_header(self):
        """test for "header" or "documentation" for first block"""
        self.converter.state = "" # normally set by the `convert` method
        self.converter.set_state([".. header", " block"])
        assert self.converter.state == "header"
        self.converter.state = "" # normally set by the `convert` method
        self.converter.set_state(["documentation", "block"])
        assert self.converter.state == "documentation"

    def test_set_state_code_block(self):
        """test for "header" or "documentation" for "code_block" """
        # normally set by the `convert` method
        self.converter._textindent = 0
        self.converter.state = "code_block"
        self.converter.set_state(["documentation", "  block"])
        assert self.converter.state == "documentation"

        self.converter.state = "code_block"
        self.converter.set_state(["  documentation", "block"])
        assert self.converter.state == "documentation"

        self.converter.state = "code_block"
        self.converter.set_state(["  code", "  block"])
        print self.converter.state
        assert self.converter.state == "code_block"

    def test_header_handler(self):
        """should strip header-string from header"""
        self.converter._codeindent = 0
        sample = [".. header", " block"]
        lines = [line for line in self.converter.header_handler(sample)]
        print lines
        assert lines == ["header", "block"]

    def test_documentation_handler(self):
        """should add comment string to documentation"""
        sample = ["doc", "block", ""]
        lines = [line for line
                  in self.converter.documentation_handler(sample)]
        print lines
        assert lines == ["# doc", "# block", "# "]

    def test_documentation_handler_set_state(self):
        """should add comment string to documentation"""
        sample = ["doc", "block::", ""]
        lines = [line for line
                  in self.converter.documentation_handler(sample)]
        print lines
        assert lines == ["# doc", "# block::", ""]
        assert self.converter.state == "code_block"

    def test_code_block_handler(self):
        """should un-indent code-blocks"""
        self.converter._codeindent = 0 # normally set in `convert`
        sample = ["  code", "  block", ""]
        lines = [line for line
                  in self.converter.code_block_handler(sample)]
        print lines
        assert lines == ["code", "block", ""]


## base tests on the "long" test data ::

    def test_call(self):
        """Calling a Text2Code instance should return the converted data as list of lines"""
        output = self.converter()
        print repr(codedata)
        print repr(output)
        assert codedata == output

    def test_call_strip(self):
        """strip=True should strip text parts"""
        self.converter.strip = True
        output = self.converter()
        print repr(stripped_code.splitlines(True))
        print repr(output)
        assert stripped_code.splitlines(True) == output

    def test_str(self):
        outstr = str(self.converter)
        print repr(code)
        print repr(outstr)
        assert code == outstr

    def test_str_strip1(self):
        """strip=True should strip text parts.

        Version 1 with `strip` given as optional argument"""
        outstr = str(Text2Code(textdata, strip=True))
        print "ist ", repr(outstr)
        print "soll", repr(stripped_code)
        # pprint(outstr)
        assert stripped_code == outstr

    def test_str_strip2(self):
        """strip=True should strip text parts

        Version 2 with `strip` set after instantiation"""
        self.converter.strip = True
        outstr = str(self.converter)
        print "ist ", repr(outstr)
        print "soll", repr(stripped_code)
        # pprint(outstr)
        assert stripped_code == outstr

    def test_malindented_code_line(self):
        """raise error if code line is less indented than code-indent"""
        data1 = ["..    #!/usr/bin/env python\n", # indent == 4 * " "
                "\n",
                "  print 'hello world'"]          # indent == 2 * " "
        data2 = ["..\t#!/usr/bin/env python\n",   # indent == 8 * " "
                "\n",
                "  print 'hello world'"]          # indent == 2 * " "
        for data in (data1, data2):
            try:
                blocks = Text2Code(data)()
                assert False, "wrong indent did not raise ValueError"
            except ValueError:
                pass

    def test_str_different_comment_string(self):
        """Convert only comments with the specified comment string to text
        """
        data = ["..  #!/usr/bin/env python\n",
                '\n',
                '::\n',  # leading code block as header
                '\n',
                "  block1 = 'first block'\n",
                '\n',
                'more text']
        soll = "\n".join(["#!/usr/bin/env python",
                          "",
                          "##::",
                          "",
                          "block1 = 'first block'",
                          "",
                          "##more text"]
                        )
        outstr = str(Text2Code(data, comment_string="##"))
        print "soll:", repr(soll)
        print "ist: ", repr(outstr)
        assert outstr == soll

    # Filters: test pre- and postprocessing of data

    def test_get_filter_preprocessor(self):
        """should return filter from filter_set for language"""
        preprocessor = self.converter.get_filter("preprocessors", "rl")
        print preprocessor
        assert preprocessor == l2r_filter

    def test_get_filter_postprocessor(self):
        """should return filter from filter_set for language"""
        postprocessor = self.converter.get_filter("postprocessors", "x")
        print postprocessor
        assert postprocessor == u2x_filter

    def test_get_css_postprocessor(self):
        """should return filter from filter_set for language"""
        postprocessor = self.converter.get_filter("postprocessors", "css")
        print postprocessor
        assert postprocessor == dumb_c_postprocessor

    def test_get_filter_nonexisting_language_filter(self):
        """should return identity_filter if language has no filter in set"""
        preprocessor = self.converter.get_filter("preprocessors", "foo")
        print preprocessor
        assert preprocessor == identity_filter

    def test_get_filter_nonexisting_filter_set(self):
        """should return identity_filter if filter_set does not exist"""
        processor = self.converter.get_filter("foo_filters", "foo")
        print processor
        assert processor == identity_filter

    def test_preprocessor(self):
        """Preprocess data with registered preprocessor for language"""
        output = Text2Code(textdata, language="x", comment_string="# ")()
        soll = [line for line in u2x_filter(codedata)]
        print "soll: ", repr(soll)
        print "ist:  ", repr(output)
        assert output == soll

    def test_postprocessor(self):
        """Preprocess data with registered postprocessor for language"""
        output = Text2Code(textdata, language="x", comment_string="# ")()
        soll = [line for line in u2x_filter(codedata)]
        print "soll:", repr(soll)
        print "ist: ", repr(output)
        assert output == soll

## Special Cases
## -------------
##
## Code follows text block without blank line
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
## End of text block detected ('::') but no paragraph separator (blank line)
## follows
##
## It is an reStructuredText syntax error, if a "literal block
## marker" is not followed by a blank line.
##
## Assuming that no double colon at end of line occurs accidentally,
## pylit could fix this and issue a warning::

# Do we need this feature? (Complicates code a lot)
# textsamples["ensure blank line after text"] = (
# """text followed by a literal block::
#   block1 = 'first block'
# """,
# """# text followed by a literal block::
#
# block1 = 'first block'
# """)

## Text follows code block without blank line
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
## End of code block detected (a line not more indented than the preceding text
## block)
##
## reStructuredText syntax demands a paragraph separator (blank line) before
## it.
##
## Assuming that the unindent is not accidental, pylit could fix this and
## issues a warning::

# Do we need this feature? (Complicates code)
# textsamples["ensure blank line after code"] = (
# """::
#
#   block1 = 'first block'
# more text
# """,
# """# ::
#
# block1 = 'first block'
#
# more text
# """)

## Options follow code-block directive
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

textsamples["code-block directive options"] = (
"""\
::
  :option: argument

  this = 'code'
""",
"""\
# ::
#   :option: argument

this = 'code'
""")

textsamples["no code-block directive options"] = (
"""\
::
  text following ``::`` without blank line

  more documentation
""",
"""\
# ::
#   text following ``::`` without blank line
# 
#   more documentation
""")

## A double colon on a line on its own
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
## As a double colon is added by the Code2Text conversion after a text block
## (if not already present), it could be removed by the Text2Code conversion
## to keep the source small and pretty.
##
## However, this would put the text and code source line numbers out of sync,
## which is bad for error reporting, failing doctests, and the JED editor
## support with the `pylit_buffer()` function in
## http://jedmodes.sf.net/mode/pylit.sl .
##
## Maybe this could be left to a post-processing filter::

# textsamples["remove single double colon"] = (
#    ["text followed by a literal block\n",
#     "\n",
#     "::\n",
#     "\n",
#     "  foo = 'first'\n"]
#    ["", # empty header
#     "# text followed by a literal block\n\n",
#     "foo = 'first'\n"]

## header samples
## ~~~~~~~~~~~~~~
## Convert a leading reStructured text comment  (variant: only if there is
## content on the first line) to a leading code block.  Return an empty list,
## if there is no header. ::

textsamples["simple header"] = ("..  print 'hello world'",
                                "print 'hello world'")

textsamples["no header (start with text)"] = (
"""a classical example without header::

  print 'hello world'
""",
"""# a classical example without header::

print 'hello world'
""")


textsamples["no header (start with blank line)"] = (
"""
a classical example without header::

  print 'hello world'
""",
"""# 
# a classical example without header::

print 'hello world'
""")


textsamples["standard header, followed by text"] = (
"""..  #!/usr/bin/env python
  # -*- coding: iso-8859-1 -*-

a classical example with header::

  print 'hello world'
""",
"""#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# a classical example with header::

print 'hello world'
""")

textsamples["standard header, followed by code"] = (
"""..  #!/usr/bin/env python

  print 'hello world'
""",
"""#!/usr/bin/env python

print 'hello world'
""")

textsamples["null string"] = ("", "", "")

## Code2Text
## =========
##
## ::

class test_Code2Text(object):

    def setUp(self):
        self.converter = Code2Text(codedata)

## Code2Text.strip_literal_marker
##
## * strip `::`-line as well as preceding blank line if on a line on its own
## * strip `::` if it is preceded by whitespace.
## * convert `::` to a single colon if preceded by text
##
## ::
    def check_strip_code_block_marker(self, sample):
        """test Code2Text.strip_code_block_marker"""
        ist = sample[0].splitlines(True)
        soll = sample[1].splitlines(True)
        print "before", ist
        converter = Code2Text(codedata)
        converter.strip_code_block_marker(ist)
        print "soll:", repr(soll)
        print "ist: ", repr(ist)
        assert ist == soll


    def test_strip_code_block_marker(self):
        samples = (("text\n\n::\n\n", "text\n\n"),
                   ("text\n::\n\n", "text\n\n"),
                   ("text ::\n\n", "text\n\n"),
                   ("text::\n\n", "text:\n\n"),
                   ("text:\n\n", "text:\n\n"),
                   ("text\n\n", "text\n\n"),
                   ("text\n", "text\n")
                   )
        for sample in samples:
            yield (self.check_strip_code_block_marker, sample)

## Code2Text.set_state
## ::

    def test_set_state(self):
        samples = (("code_block", ["code_block\n"], "code_block"),
                   ("code_block", ["#code_block\n"], "code_block"),
                   ("code_block", ["## code_block\n"], "code_block"),
                   ("code_block", ["# documentation\n"], "documentation"),
                   ("code_block", ["#  documentation\n"], "documentation"),
                   ("code_block", ["# \n"], "documentation"),
                   ("code_block", ["#\n"], "documentation"),
                   ("code_block", ["\n"], "documentation"),
                   ("", ["code_block\n"], "header"),
                   ("", ["# documentation\n"], "documentation"),
                   ("documentation", ["code_block\n"], "code_block"),
                   ("documentation", ["# documentation\n"], "documentation"),
                  )
        print "comment string", repr(self.converter.comment_string)
        for (old_state, lines, soll) in samples:
            self.converter.state = old_state
            self.converter.set_state(lines)
            print repr(lines), "old state", old_state
            print "soll", repr(soll),
            print "result", repr(self.converter.state)
            assert soll == self.converter.state

## base tests on the "long" test strings ::

    def test_call(self):
        output = self.converter()
        print repr(textdata)
        print repr(output)
        assert textdata == output

    def test_call_strip(self):
        output = Code2Text(codedata, strip=True)()
        print repr(stripped_text.splitlines(True))
        print repr(output)
        assert stripped_text.splitlines(True) == output

    def test_str(self):
        """Test Code2Text class converting code->text"""
        outstr = str(self.converter)
        # print text
        print "soll:", repr(text)
        print "ist: ", repr(outstr)
        assert text == outstr

    def test_str_strip(self):
        """Test Code2Text class converting code->rst with strip=True

        Should strip code blocks
        """
        outstr = str(Code2Text(codedata, strip=True))
        print repr(stripped_text)
        print repr(outstr)
        assert stripped_text == outstr

    def test_str_different_comment_string(self):
        """Convert only comments with the specified comment string to text
        """
        outstr = str(Code2Text(codedata, comment_string="##", strip=True))
        print outstr
        assert outstr == ""
        data = ["# ::\n",
                "\n",
                "block1 = 'first block'\n",
                "\n",
                "## more text"]
        soll = "\n".join(['..  # ::',  # leading code block as header
                          '  ',
                          "  block1 = 'first block'",
                          '  ',
                          ' more text']   # keep space (not part of comment string)
                        )
        outstr = str(Code2Text(data, comment_string="##"))
        print "soll:", repr(soll)
        print "ist: ", repr(outstr)
        assert outstr == soll

    def test_call_different_code_block_marker(self):
        """recognize specified code-block marker
        """
        data = ["# .. code-block:: python3\n",
                "\n",
                "block1 = 'first block'\n",
                "\n",
                "#  more text\n"]
        soll = ['.. code-block:: python3\n',
                '\n',
                "  block1 = 'first block'\n",
                '  \n',
                ' more text\n']   # keep space (not part of comment string)

        converter = Code2Text(data, code_block_marker='.. code-block::')
        output = converter()
        print "soll:", repr(soll)
        print "ist: ", repr(output)
        assert output == soll

    # Filters: test pre- and postprocessing of Code2Text data conversion

    def test_get_filter_preprocessor(self):
        """should return Code2Text preprocessor for language"""
        preprocessor = self.converter.get_filter("preprocessors", "rl")
        print preprocessor
        assert preprocessor == r2l_filter

    def test_get_css_preprocessor(self):
        """should return filter from filter_set for language"""
        preprocessor = self.converter.get_filter("preprocessors", "css")
        print preprocessor
        assert preprocessor == dumb_c_preprocessor

    def test_get_filter_postprocessor(self):
        """should return Code2Text postprocessor for language"""
        postprocessor = self.converter.get_filter("postprocessors", "x")
        print postprocessor
        assert postprocessor == x2u_filter

    def test_get_filter_nonexisting_language_filter(self):
        """should return identity_filter if language has no filter in set"""
        preprocessor = self.converter.get_filter("preprocessors", "foo")
        print preprocessor
        assert preprocessor == identity_filter

    def test_get_filter_nonexisting_filter_set(self):
        """should return identity_filter if filter_set does not exist"""
        processor = self.converter.get_filter("foo_filters", "foo")
        print processor
        assert processor == identity_filter

    def test_preprocessor(self):
        """Preprocess data with registered preprocessor for language"""
        converter = Code2Text(codedata, language="rl", comment_string="# ")
        print "preprocessor", converter.preprocessor
        print "postprocessor", converter.postprocessor
        output = converter()
        soll = [line.replace("r", "l") for line in textdata]
        print "ist: ", repr(output)
        print "soll:", repr(soll)
        assert output == soll

    def test_postprocessor(self):
        """Postprocess data with registered postprocessor for language"""
        output = Code2Text(codedata, language="x", comment_string="# ")()
        soll = [line.replace("x", "u") for line in textdata]
        print "soll:", repr(soll)
        print "ist: ", repr(output)
        assert output == soll


## Special cases
## -------------
##
## blank comment line
## ~~~~~~~~~~~~~~~~~~
##
## Normally, whitespace in the comment string is significant, i.e. with
## ``comment_string = "# "``, a line ``"#something\n"`` will count as code.
##
## However, if a comment line is blank, trailing whitespace in the comment
## string should be ignored, i.e. ``#\n`` is recognised as a blank text line::

codesamples["ignore trailing whitespace in comment string for blank line"] = (
"""# ::

block1 = 'first block'

#
# more text
""",
"""::

  block1 = 'first block'
  

more text
""")

## No blank line after text
## ~~~~~~~~~~~~~~~~~~~~~~~~
##
## If a matching comment precedes or follows a code line (i.e. any line
## without matching comment) without a blank line in between, it counts as code
## line.
##
## This will keep small inline comments close to the code they comment on. It
## will also keep blocks together where one commented line does not match the
## comment string (the whole block will be kept as commented code)
## ::

codesamples["comment before code (without blank line)"] = (
"""\
# this is text::

# this is a comment
foo = 'first'
""",
"""\
this is text::

  # this is a comment
  foo = 'first'
""",
"""\
this is text:

""")

codesamples["comment block before code (without blank line)"] = (
"""\
# no text (watch the comment sign in the next line)::
#
# this is a comment
foo = 'first'
""",
"""\
..  # no text (watch the comment sign in the next line)::
  #
  # this is a comment
  foo = 'first'
""",
"")

codesamples["comment after code (without blank line)"] = (
"""\
# ::

block1 = 'first block'
# commented code

# text again
""",
"""\
::

  block1 = 'first block'
  # commented code
  
text again
""",
"""
text again
""")

codesamples["comment block after code (without blank line)"] = (
"""\
# ::

block1 = 'first block'
# commented code
#
# still comment
""",
"""::

  block1 = 'first block'
  # commented code
  #
  # still comment
""",
"""
""")

## missing literal block marker
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
## If text (with matching comment string) is followed by code (line(s) without
## matching comment string), but there is no double colon at the end, back
## conversion would not recognise the end of text!
##
## Therefore, pylit adds a paragraph containing only ``::`` -- the literal
## block marker in expanded form. (While it would in many cases be nicer to
## add the double colon to the last text line, this is not always valid rst
## syntax, e.g. after a section header or a list. Therefore the automatic
## insertion will use the save form, feel free to correct this by hand.)::

codesamples["insert missing double colon after text block"] = (
"""# text followed by code without double colon

foo = 'first'
""",
"""text followed by code without double colon

::

  foo = 'first'
""",
"""text followed by code without double colon

""")

codesamples["ignore directive options when looking for code-block marker"] = (
"""\
# ::
#   :option: argument
#   :option2: argument

this = 'code'
""",
"""\
::
  :option: argument
  :option2: argument

  this = 'code'
""")

codesamples["code-block marker followed by text not a directive option"] = (
"""\
# ::
#   text following ``::`` without blank line

this = 'code'
""",
"""\
::
  text following ``::`` without blank line

::

  this = 'code'
""")


## header samples
## ~~~~~~~~~~~~~~
##
## Convert a header (leading code block) to a reStructured text comment. ::

codesamples["no matching comment, just code"] = (
"""print 'hello world'

print 'ende'
""",
"""..  print 'hello world'
  
  print 'ende'
""")

codesamples["empty header (start with matching comment)"] = (
"""# a classical example without header::

print 'hello world'
""",
"""a classical example without header::

  print 'hello world'
""",
"""a classical example without header:

""")

codesamples["standard header, followed by text"] = (
"""#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# a classical example with header::

print 'hello world'
""",
"""..  #!/usr/bin/env python
  # -*- coding: iso-8859-1 -*-
  
a classical example with header::

  print 'hello world'
""",
"""a classical example with header:

""")

codesamples["standard header, followed by code"] = (
"""#!/usr/bin/env python

print 'hello world'
""",
"""..  #!/usr/bin/env python
  
  print 'hello world'
""",
"")

## Filter tests
## ============
##
## ::

css_code = ['/* import the default Docutils style sheet */\n',
            '/* --------------------------------------- */\n',
            '\n',
            '/* :: */\n',
            '\n',
            '/*comment*/\n',
            '@import url("html4css1.css"); /* style */\n']

## ::

css_filtered_code = ['// import the default Docutils style sheet\n',
                     '// ---------------------------------------\n',
                     '\n',
                     '// ::\n',
                     '\n',
                     '/*comment*/\n',
                     '@import url("html4css1.css"); /* style */\n']

## ::

def test_dumb_c_preprocessor():
    """convert `C` to `C++` comments"""
    output = [line for line in dumb_c_preprocessor(css_code)]
    print "ist:  %r"%output
    print "soll: %r"%css_filtered_code
    assert output == css_filtered_code

## ::

def test_dumb_c_postprocessor():
    """convert `C++` to `C` comments"""
    output = [line for line in dumb_c_postprocessor(css_filtered_code)]
    print "ist:  %r"%output
    print "soll: %r"%css_code
    assert output == css_code



## ::

if __name__ == "__main__":
    nose.runmodule() # requires nose 0.9.1
    sys.exit()
