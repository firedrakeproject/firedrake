from __future__ import annotations

import abc
import collections
import functools
import itertools
import weakref
from collections.abc import Hashable
from typing import Any, Union

from immutabledict import immutabledict as idict

import pyop3.obj
from pyop3 import collections as op3_collections
from pyop3 import utils
from pyop3.collections import OrderedFrozenSet


def postorder(method):
    """Postorder decorator.

    It is more natural for users to write a post-order singledispatchmethod
    whose arguments are ``(self, o, *processed_operands, **kwargs)``,
    while `DAGTraverser` expects one whose arguments are
    ``(self, o, **kwargs)``.
    This decorator takes the former and converts to the latter, processing
    ``o.ufl_operands`` behind the users.

    """
    @functools.wraps(method)
    def _postorder_node(self, node, **kwargs):
        new_children = {}
        for attr_name, child_attr in self.children(node).items():
            if isinstance(child_attr, tuple):
                new_children[attr_name] = tuple(
                    self(item, **kwargs)
                    for item in child_attr
                )
            elif isinstance(child_attr, idict):
                new_children[attr_name] = idict({
                    key: self(value, **kwargs)
                    for key, value in child_attr.items()
                })
            else:
                new_children[attr_name] = self(child_attr, **kwargs)
        new_children = idict(new_children)
        return method(self, node, new_children, **kwargs)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if isinstance(self, NodeVisitor):
            return _postorder_node(self, *args, **kwargs)
        else:
            raise TypeError(f"Cannot postorder visit '{utils.pretty_type(self)}'")

    return wrapper


# maybe implement record_new here?
class Node(pyop3.obj.Object, abc.ABC):
    # bikeshedding, since this is meant to be inherited from it would be good to 'namespace' it
    @property
    @abc.abstractmethod
    def child_attrs(self):
        pass

    @property
    def children(self) -> idict:
        return idict({attr: getattr(self, attr) for attr in self.child_attrs})


class Operator(Node):
    """A non-terminal."""


class Terminal(Node, abc.ABC):
    child_attrs = ()


"""Taken from UFL"""
class Visitor(abc.ABC):
    """Base class for DAG traversers.

    Args:
        compress: If True, ``result_cache`` will be used.
        visited_cache: cache of intermediate results; expr -> r = self.process(expr, ...).
        result_cache: cache of result objects for memory reuse, r -> r.

    """

    def __init__(
        self,
        *,
        reuse_results: bool = True,
    ) -> None:
        if reuse_results:
            result_cache = {}
        else:
            result_cache = None

        self.reuse_results = reuse_results

        self._visited_cache = {}
        self._result_cache = result_cache
        self.index = ()
        self._index_stack = collections.defaultdict(itertools.count)

    # {{{ overrideable interface

    def __call__(self, node, *args, **kwargs):
        """Perform memoised DAG traversal with ``process`` singledispatch method.

        Args:
            node:
                Expression to start DAG traversal from.
            *args:
                Positional arguments for the ``process`` singledispatchmethod.
            **kwargs:
                keyword arguments for the ``process`` singledispatchmethod.

        Returns:
            Processed Expression.

        """
        try:
            # Push the current index (location in the tree) onto a stack
            prev_index = self.index
            self.index += (next(self._index_stack[self.index]),)

            if isinstance(node, weakref.ReferenceType):
                # Don't try to cache weakrefs
                node = node()
                assert node is not None, "Cannot do a traversal involving a dead weakref"
                do_cache = False
            else:
                cache_key = self.get_cache_key(node, *args, **kwargs)
                if cache_key in self._visited_cache:
                    return self._visited_cache[cache_key]
                do_cache = True

            result = self.process(node, *args, **kwargs)
            # Conditionally check if r is in result_cache, a memory optimization
            # to be able to keep representation of result compact
            if do_cache and self.reuse_results:
                try:
                    # Cache hit: Use previously computed object, allowing current
                    # ``result`` to be garbage collected as soon as possible
                    result = self._result_cache[result]
                except KeyError:
                    # Cache miss: store in result_cache
                    self._result_cache[result] = result

            # Store result in cache
            if do_cache:
                self._visited_cache[cache_key] = result
            return result
        finally:
            self.index = prev_index

    def get_cache_key(self, node, *args, **kwargs) -> Hashable:
        """Maybe overload this if you want to set some things up"""
        return (node, args, tuple((k, v) for k, v in kwargs.items()))

    # TODO: Make this 'process_invalid' and make process an abstract method
    def process(self, o: Expr, **kwargs) -> Expr:
        """Process node by type.

        Args:
            o:
                UFL expression to start DAG traversal from.
            **kwargs:
                Keyword arguments for the ``process`` singledispatchmethod.

        Returns:
            Processed :py:class:`Expr`.
        """
        raise TypeError(f"'{utils.pretty_type(self)}' does not define a rule for '{utils.pretty_type(o)}'")

    # }}}

class NodeVisitor(Visitor):

    @functools.singledispatchmethod
    def children(self, node, /):
        raise TypeError(f"{utils.pretty_type(node)} not recognised")

    @children.register(Node)
    def _(self, node, /):
        return node.children


class NodeTransformer(NodeVisitor, abc.ABC):

    @postorder
    def reuse_if_untouched(self, node: Node, visited, **kwargs) -> Node:
        """Reuse if untouched.

        Args:
            o:
                Expression to start DAG traversal from.
            **kwargs:
                Keyword arguments for the ``process`` singledispatchmethod.

        Returns:
            Processed expression.

        """
        if all(
            getattr(node, attr_name) == attr
            for attr_name, attr in visited.items()
        ):
            return node
        else:
            return node.record_new(**visited)


class NodeCollector(NodeVisitor, abc.ABC):

    @functools.singledispatchmethod
    def process(self, obj: Any, /) -> OrderedFrozenSet:
        return super().process(obj)

    @process.register(tuple)
    @postorder
    def _(self, tuple_, visited, /) -> OrderedFrozenSet:
        return OrderedFrozenSet().union(*visited.values())
