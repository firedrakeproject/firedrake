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

from abc import ABCMeta, abstractmethod

from tsfc.node import Node as NodeBase


class Node(NodeBase):
    """Base class of all Impero nodes"""

    __slots__ = ()


class Terminal(Node):
    """Abstract class for terminal Impero nodes"""

    __metaclass__ = ABCMeta

    __slots__ = ()

    children = ()

    @abstractmethod
    def loop_shape(self, free_indices):
        """Gives the loop shape, an ordering of indices for an Impero
        terminal.

        :arg free_indices: a callable mapping of GEM expressions to
                           ordered free indices.
        """
        pass


class Evaluate(Terminal):
    """Assign the value of a GEM expression to a temporary."""

    __slots__ = ('expression',)
    __front__ = ('expression',)

    def __init__(self, expression):
        self.expression = expression

    def loop_shape(self, free_indices):
        return free_indices(self.expression)


class Initialise(Terminal):
    """Initialise an :class:`gem.IndexSum`."""

    __slots__ = ('indexsum',)
    __front__ = ('indexsum',)

    def __init__(self, indexsum):
        self.indexsum = indexsum

    def loop_shape(self, free_indices):
        return free_indices(self.indexsum)


class Accumulate(Terminal):
    """Accumulate terms into an :class:`gem.IndexSum`."""

    __slots__ = ('indexsum',)
    __front__ = ('indexsum',)

    def __init__(self, indexsum):
        self.indexsum = indexsum

    def loop_shape(self, free_indices):
        return free_indices(self.indexsum.children[0])


class Noop(Terminal):
    """No-op terminal. Does not generate code, but wraps a GEM
    expression to have a loop shape, thus affects loop fusion."""

    __slots__ = ('expression',)
    __front__ = ('expression',)

    def __init__(self, expression):
        self.expression = expression

    def loop_shape(self, free_indices):
        return free_indices(self.expression)


class Return(Terminal):
    """Save value of GEM expression into an lvalue. Used to "return"
    values from a kernel."""

    __slots__ = ('variable', 'expression')
    __front__ = ('variable', 'expression')

    def __init__(self, variable, expression):
        assert set(variable.free_indices) >= set(expression.free_indices)

        self.variable = variable
        self.expression = expression

    def loop_shape(self, free_indices):
        return free_indices(self.variable)


class ReturnAccumulate(Terminal):
    """Accumulate an :class:`gem.IndexSum` directly into a return
    variable."""

    __slots__ = ('variable', 'indexsum')
    __front__ = ('variable', 'indexsum')

    def __init__(self, variable, indexsum):
        assert set(variable.free_indices) == set(indexsum.free_indices)

        self.variable = variable
        self.indexsum = indexsum

    def loop_shape(self, free_indices):
        return free_indices(self.indexsum.children[0])


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
