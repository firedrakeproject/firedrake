from __future__ import annotations

import abc
import functools
from collections.abc import Hashable
from functools import cached_property
from typing import Any

from immutabledict import immutabledict as idict
from pyop3 import utils
from pyop3.utils import OrderedFrozenSet


# maybe implement __record_init__ here?
class Node(abc.ABC):
    # bikeshedding, since this is meant to be inherited from it would be good to 'namespace' it
    @property
    @abc.abstractmethod
    def child_attrs(self):
        pass

    @property
    def children(self) -> idict:
        return idict({attr: getattr(self, attr) for attr in self.child_attrs})


class Terminal(Node, abc.ABC):
    child_attrs = ()


# TODO: Add mixin for stateless transformers...
# def maybe_singleton():  cached on the mesh (or outermost 'heavy' cache)


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
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr] | None = None,
        result_cache: dict[Expr, Expr] | None = None,
    ) -> None:
        """Initialise."""
        self._compress = compress
        self._visited_cache = {} if visited_cache is None else visited_cache
        self._result_cache = {} if result_cache is None else result_cache
        self._seen_keys = set()

    # {{{ overrideable interface

    def __call__(self, *args, **kwargs):
        """Maybe overload this if you want to set some things up"""
        return self._call(*args, **kwargs)

    def get_cache_key(self, node, **kwargs) -> Hashable:
        """Maybe overload this if you want to set some things up"""
        return (node, tuple((k, v) for k, v in kwargs.items()))

    def preprocess_node(self, node) -> tuple[Any, ...]:
        return (node,)

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
        raise AssertionError(f"'{utils.pretty_type(self)}' does not define a rule for '{utils.pretty_type(o)}'")

    @staticmethod
    def postorder(method):
        """Postorder decorator.

        It is more natural for users to write a post-order singledispatchmethod
        whose arguments are ``(self, o, *processed_operands, **kwargs)``,
        while `DAGTraverser` expects one whose arguments are
        ``(self, o, **kwargs)``.
        This decorator takes the former and converts to the latter, processing
        ``o.ufl_operands`` behind the users.

        """
        raise NotImplementedError

    # }}}

    def _call(self, node: Expr, **kwargs) -> Expr:
        """Perform memoised DAG traversal with ``process`` singledispatch method.

        Args:
            node:
                Expression to start DAG traversal from.
            **kwargs:
                keyword arguments for the ``process`` singledispatchmethod.

        Returns:
            Processed Expression.

        """
        cache_key = self.get_cache_key(node, **kwargs)
        try:
            return self._visited_cache[cache_key]
        except KeyError:
            self._seen_keys.add(cache_key)
            preprocessed = self.preprocess_node(node)
            result = self.process(*preprocessed, **kwargs)
            # Optionally check if r is in result_cache, a memory optimization
            # to be able to keep representation of result compact
            if self._compress:
                try:
                    # Cache hit: Use previously computed object, allowing current
                    # ``result`` to be garbage collected as soon as possible
                    result = self._result_cache[result]
                except KeyError:
                    # Cache miss: store in result_cache
                    self._result_cache[result] = result
            # Store result in cache
            self._visited_cache[cache_key] = result
            return result


class LabelledTreeVisitor(Visitor):
    """
    Notes
    -----
    Empty or unit trees get passed `None`.

    """

    def __init__(self):
        super().__init__()

        # variables that are only valid mid traversal
        self._tree = None

    # {{{ abstract methods

    @property
    @staticmethod
    @abc.abstractmethod
    def EMPTY():
        pass

    # }}}

    # {{{ interface impls

    def __call__(self, tree: AxisTree, **kwargs):
        try:
            self._tree = tree
            return self._call(idict(), **kwargs)
        finally:
            self._tree = None

    def get_cache_key(self, path: ConcretePathT, **kwargs) -> Hashable:
        # an axis is uniquely identified by itself and the subtree beneath it
        return (
            self._tree._subtree_node_map(path),
            tuple((k, v) for k, v in kwargs.items()),
        )

    def preprocess_node(self, path: ConcetePathT, /) -> tuple[TreeNode, ConcretePathT]:
        return (self._tree.node_map[path], path)

    @staticmethod
    def postorder(method):
        @functools.wraps(method)
        def wrapper(self, node, path, **kwargs):
            visited = []
            for component_label in node.component_labels:
                path_ = path | {node.label: component_label}
                if self._tree.node_map[path_]:
                    visited.append(self._call(path_, **kwargs))
                else:
                    visited.append(self.EMPTY)
            visited = tuple(visited)
            return method(self, node, path, visited, **kwargs)
        return wrapper

    # }}}


class NodeVisitor(Visitor):

    @staticmethod
    def postorder(method):
        @functools.wraps(method)
        def wrapper(self, node, **kwargs):
            new_children = {}
            for attr_name, child_attr in node.children.items():
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
        return wrapper


class NodeTransformer(NodeVisitor, abc.ABC):
    def reuse_if_untouched(self, o: Expr | BaseForm, **kwargs) -> Expr | BaseForm:
        """Reuse if untouched.

        Args:
            o:
                Expression to start DAG traversal from.
            **kwargs:
                Keyword arguments for the ``process`` singledispatchmethod.

        Returns:
            Processed expression.

        """
        raise NotImplementedError("handle tuples etc")
        new_children = idict({
            name: self(child, **kwargs)
            for name, child in o.children.items()
        })
        if new_children == o.children:
            return o
        else:
            return o.__record_init__(**new_children)


class NodeCollector(NodeVisitor, abc.ABC):

    @functools.singledispatchmethod
    def process(self, obj: Any, /) -> OrderedFrozenSet:
        return super().process(obj)

    @process.register(tuple)
    @NodeVisitor.postorder
    def _(self, tuple_, visited, /) -> OrderedFrozenSet:
        return OrderedFrozenSet().union(*visited.values())
