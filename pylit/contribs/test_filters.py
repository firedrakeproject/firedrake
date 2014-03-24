# Tests
# =====
#

from filters import *


# comment string for Elisp
COMMENT_STRING = ';; '

# simulate calling from pylit main
keyw = { 'comment_string':COMMENT_STRING }


# Auxiliary functions
# ===================
#

def test(actual_iter, expected_iter):
    """Compare all lines in `actual_iter` with lines in `expected_iter`"""
    actual = []
    for line in actual_iter:
        actual.append(line)
    expected = []
    for line in expected_iter:
        expected.append(line)
    if len(actual) == len(expected):
        result = True
    else:
        result = False
    for i in range(0, min(len(actual),len(expected))):
        if actual[i] != expected[i]:
            result = False
    if result:
        return "OK"
    else:
        return "Failed (expected: %s, got: %s)" \
               % (expected, actual)


# Test fixtures
# =============
#

sc_in = IdentityFilter([
    ";; This should be stripped.\n",
    "This should be not.\n"
    ])
sc_out = IdentityFilter([
    "This should be stripped.\n",
    "This should be not.\n"
    ])

blank_lines = IdentityFilter([
    "\n",
    "\n",
    "\n"
    ])

c2t_in_1 = IdentityFilter([
    ";;;Commentary:\n"
    ])
c2t_out_1 = IdentityFilter([
    ";; .. |elisp> ;;;Commentary:\n",
    ])
c2t_in_2 = IdentityFilter([
    ";;;Commentary:\n",
    ";; This is\n",
    ";; a test."
    ])
c2t_out_2 = IdentityFilter([
    ";; .. |elisp> ;;;Commentary:\n",
    ";; This is\n",
    ";; a test."
    ])


# Test cases
# ==========
#

# test ElispCode2TextFilter with one sample line
f = ElispCode2TextFilter(c2t_in_1, **keyw)
print "ElispCode2TextFilter with sample line: %s" \
      % test(f, c2t_out_1)

# test ElispText2CodeFilter with one sample line
f = ElispText2CodeFilter(c2t_out_1, **keyw)
print "ElispText2CodeFilter with sample line: %s" \
      % test(f, c2t_in_1)

# test ElispCode2TextFilter with one sample para
f = ElispCode2TextFilter(c2t_in_2, **keyw)
print "ElispCode2TextFilter with sample paragraph: %s" \
      % test(f, c2t_out_2)

# test ElispText2CodeFilter with one sample para
f = ElispText2CodeFilter(c2t_out_2, **keyw)
print "ElispText2CodeFilter with sample paragraph: %s" \
      % test(f, c2t_in_2)

# round-trip test code->txt->code with sample line
f = ElispText2CodeFilter(ElispCode2TextFilter(c2t_in_1, **keyw), **keyw)
print "round-trip test code->txt->code with sample line: %s" \
      % test(f, c2t_in_1)

# round-trip test code->txt->code with sample line
f = ElispCode2TextFilter(ElispText2CodeFilter(c2t_out_1, **keyw), **keyw)
print "round-trip test txt->code->txt with sample line: %s" \
      % test(f, c2t_out_1)

# DISABLED 2007-02-21 -- these need the reverse of strip_comments(),
# that is a function that prefix comment_string to non-code lines.
# But that would be duplicating PyLit's code here, so just forget it.
#
# round-trip test code->txt->code with sample para
#f = ElispText2CodeFilter(IdentityFilter(ElispCode2TextFilter(c2t_in_2,
#                                                           **keyw)), **keyw)
#print "round-trip test code->txt->code with sample paragraph: %s" \
#      % test(f, c2t_in_2)
#
# round-trip test code->txt->code with sample para
#f = strip_comments(ElispCode2TextFilter(
#    strip_indent(ElispText2CodeFilter(c2t_out_2, **keyw)), **keyw))
#print "round-trip test txt->code->txt with sample paragraph: %s" \
#      % test(f, c2t_out_2)

# txt->code should preserve blank lines
f = ElispText2CodeFilter(blank_lines, **keyw)
print "txt->code with blank lines: %s" \
      % test(f, blank_lines)

# code->txt should preserve blank lines
f = ElispCode2TextFilter(blank_lines, **keyw)
print "code->txt with blank lines: %s" \
      % test(f, blank_lines)

# round-trip test txt->code->txt
f = ElispText2CodeFilter(ElispCode2TextFilter(blank_lines, **keyw), **keyw)
print "round-trip test txt->code->txt with blank lines: %s" \
      % test(f, blank_lines)

# round-trip test code->txt->code
f = ElispCode2TextFilter(ElispText2CodeFilter(blank_lines, **keyw), **keyw)
print "round-trip test txt->code->txt with blank lines: %s" \
      % test(f, blank_lines)
