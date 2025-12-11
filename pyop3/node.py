from __future__ import annotations

import abc
from collections.abc import Hashable
from functools import cached_property, singledispatchmethod, wraps

from immutabledict import immutabledict as idict


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

    @property
    def is_final(self) -> bool:
        return False

    @property
    def _disk_cache_key(self) -> Hashable:
        raise NotImplementedError(f"{type(self).__name__} does not define a '_disk_cache_key' attribute")

    @cached_property
    def disk_cache_key(self) -> Hashable:
        """Key used to write the element to disk.

        The returned key should be consistent across ranks and not include
        overly specific information such as buffer names or array values.

        For mutable objects like buffers, the key should renumber them from 0
        to achieve maximal reuse.

        Lowered operations consisting solely of codegen-ready objects can use a
        disk cache. This is because the type guarantees that no further
        data-dependent transformations will occur.

        """
        assert self.is_final
        return self._disk_cache_key


class Terminal(Node, abc.ABC):
    child_attrs = ()


"""Taken from UFL"""
class DAGTraverser(abc.ABC):
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

    def __call__(self, node: Expr, **kwargs) -> Expr:
        """Perform memoised DAG traversal with ``process`` singledispatch method.

        Args:
            node:
                Expression to start DAG traversal from.
            **kwargs:
                keyword arguments for the ``process`` singledispatchmethod.

        Returns:
            Processed Expression.

        """
        cache_key = (node, tuple((k, v) for k, v in kwargs.items()))
        try:
            return self._visited_cache[cache_key]
        except KeyError:
            result = self._safe_process(node, **kwargs)
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

    @singledispatchmethod
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
        raise AssertionError(f"Rule not set for {type(o)}")

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
        new_children = {
            name: self(child, **kwargs)
            for name, child in o.children
        }
        if new_children == o.children:
            return o
        else:
            return o.__record_init__(**new_children)

def postorder(method):
    """Postorder decorator.

    It is more natural for users to write a post-order singledispatchmethod
    whose arguments are ``(self, o, *processed_operands, **kwargs)``,
    while `DAGTraverser` expects one whose arguments are
    ``(self, o, **kwargs)``.
    This decorator takes the former and converts to the latter, processing
    ``o.ufl_operands`` behind the users.

    """

    @wraps(method)
    def wrapper(self, o, **kwargs):
        processed_operands = [self(operand, **kwargs) for operand in o.ufl_operands]
        return method(self, o, *processed_operands, **kwargs)

    return wrapper


class Transformer(DAGTraverser, abc.ABC):
    def _safe_process(self, o, **kwargs):
        assert not o.is_final
        return self.process(o, **kwargs)
