"""Impero is a helper AST for generating C code (or equivalent,
e.g. COFFEE) from GEM.  An Impero expression is a proper tree, not
directed acyclic graph (DAG).  Impero is a helper AST, not a
standalone language; it is incomplete without GEM as its terminals
refer to nodes from GEM expressions.

Trivia:
 - Impero helps translating GEM into an imperative language.
 - Byzantine units in Age of Empires II sometimes say 'Impero?'
   (Command?) after clicking on them.
"""

from __future__ import absolute_import

from tsfc.node import Node as NodeBase


class Node(NodeBase):
    """Base class of all Impero nodes"""

    __slots__ = ()


class Terminal(Node):
    """Abstract class for terminal Impero nodes"""

    __slots__ = ()

    children = ()


class Evaluate(Terminal):
    """Assign the value of a GEM expression to a temporary."""

    __slots__ = ('expression',)
    __front__ = ('expression',)

    def __init__(self, expression):
        self.expression = expression


class Initialise(Terminal):
    """Initialise an :class:`gem.IndexSum`."""

    __slots__ = ('indexsum',)
    __front__ = ('indexsum',)

    def __init__(self, indexsum):
        self.indexsum = indexsum


class Accumulate(Terminal):
    """Accumulate terms into an :class:`gem.IndexSum`."""

    __slots__ = ('indexsum',)
    __front__ = ('indexsum',)

    def __init__(self, indexsum):
        self.indexsum = indexsum


class Return(Terminal):
    """Save value of GEM expression into an lvalue. Used to "return"
    values from a kernel."""

    __slots__ = ('variable', 'expression')
    __front__ = ('variable', 'expression')

    def __init__(self, variable, expression):
        self.variable = variable
        self.expression = expression


class Block(Node):
    """An ordered set of Impero expressions.  Corresponds to a curly
    braces block in C."""

    __slots__ = ('children',)

    def __init__(self, statements):
        self.children = tuple(statements)


class For(Node):
    """For loop with an index which stores its extent, and a loop body
    expression which is usually a :class:`Block`."""

    __slots__ = ('index', 'children')
    __front__ = ('index',)

    def __init__(self, index, statement):
        self.index = index
        self.children = (statement,)
