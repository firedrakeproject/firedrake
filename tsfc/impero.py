from __future__ import absolute_import

from tsfc.node import Node as node_Node


class Node(node_Node):
    __slots__ = ()


class Terminal(Node):
    __slots__ = ()

    children = ()


class Evaluate(Terminal):
    __slots__ = ('expression',)
    __front__ = ('expression',)

    def __init__(self, expression):
        self.expression = expression


class Initialise(Terminal):
    __slots__ = ('indexsum',)
    __front__ = ('indexsum',)

    def __init__(self, indexsum):
        self.indexsum = indexsum


class Accumulate(Terminal):
    __slots__ = ('indexsum',)
    __front__ = ('indexsum',)

    def __init__(self, indexsum):
        self.indexsum = indexsum


class Return(Terminal):
    __slots__ = ('variable', 'expression')
    __front__ = ('variable', 'expression')

    def __init__(self, variable, expression):
        self.variable = variable
        self.expression = expression


class Block(Node):
    __slots__ = ('children',)

    def __init__(self, statements):
        self.children = tuple(statements)


class For(Node):
    __slots__ = ('index', 'children')
    __front__ = ('index',)

    def __init__(self, index, statement):
        self.index = index
        self.children = (statement,)
