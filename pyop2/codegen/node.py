"""Generic abstract node class and utility functions for creating
expression DAG languages."""

import collections


class Node(object):
    """Abstract node class.

    Nodes are not meant to be modified.

    A node can reference other nodes; they are called children. A node
    might contain data, or reference other objects which are not
    themselves nodes; they are not called children.

    Both the children (if any) and non-child data (if any) are
    required to create a node, or determine the equality of two
    nodes. For reconstruction, however, only the new children are
    necessary.
    """

    __slots__ = ('hash_value',)

    # Non-child data as the first arguments of the constructor.
    # To be (potentially) overridden by derived node classes.
    __front__ = ()

    # Non-child data as the last arguments of the constructor.
    # To be (potentially) overridden by derived node classes.
    __back__ = ()

    def _cons_args(self, children):
        """Constructs an argument list for the constructor with
        non-child data from 'self' and children from 'children'.

        Internally used utility function.
        """
        front_args = [getattr(self, name) for name in self.__front__]
        back_args = [getattr(self, name) for name in self.__back__]

        return tuple(front_args) + tuple(children) + tuple(back_args)

    def __reduce__(self):
        # Gold version:
        return type(self), self._cons_args(self.children)

    def reconstruct(self, *args):
        """Reconstructs the node with new children from
        'args'. Non-child data are copied from 'self'.

        Returns a new object.
        """
        return type(self)(*self._cons_args(args))

    def __repr__(self):
        cons_args = self._cons_args(self.children)
        return "%s(%s)" % (type(self).__name__, ", ".join(map(repr, cons_args)))

    def __eq__(self, other):
        """Provides equality testing with quick positive and negative
        paths based on :func:`id` and :meth:`__hash__`.
        """
        if self is other:
            return True
        elif hash(self) != hash(other):
            return False
        else:
            return self.is_equal(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """Provides caching for hash values."""
        try:
            return self.hash_value
        except AttributeError:
            self.hash_value = self.get_hash()
            return self.hash_value

    def is_equal(self, other):
        """Equality predicate.

        This is the method to potentially override in derived classes,
        not :meth:`__eq__` or :meth:`__ne__`.
        """
        if type(self) != type(other):
            return False
        self_consargs = self._cons_args(self.children)
        other_consargs = other._cons_args(other.children)
        return self_consargs == other_consargs

    def get_hash(self):
        """Hash function.

        This is the method to potentially override in derived classes,
        not :meth:`__hash__`.
        """
        return hash((type(self),) + self._cons_args(self.children))


def pre_traversal(expression_dags):
    """Pre-order traversal of the nodes of expression DAGs."""
    seen = set()
    lifo = []
    # Some roots might be same, but they must be visited only once.
    # Keep the original ordering of roots, for deterministic code
    # generation.
    for root in expression_dags:
        if root not in seen:
            seen.add(root)
            lifo.append(root)

    while lifo:
        node = lifo.pop()
        yield node
        for child in reversed(node.children):
            if child not in seen:
                seen.add(child)
                lifo.append(child)


def post_traversal(expression_dags):
    """Post-order traversal of the nodes of expression DAGs."""
    seen = set()
    lifo = []
    # Some roots might be same, but they must be visited only once.
    # Keep the original ordering of roots, for deterministic code
    # generation.
    for root in expression_dags:
        if root not in seen:
            seen.add(root)
            lifo.append((root, list(root.children)))

    while lifo:
        node, deps = lifo[-1]
        for i, dep in enumerate(deps):
            if dep is not None and dep not in seen:
                lifo.append((dep, list(dep.children)))
                deps[i] = None
                break
        else:
            yield node
            seen.add(node)
            lifo.pop()


# Default to the more efficient pre-order traversal
traversal = pre_traversal


def collect_refcount(expression_dags):
    """Collects reference counts for a multi-root expression DAG."""
    result = collections.Counter(expression_dags)
    for node in traversal(expression_dags):
        result.update(node.children)
    return result


def noop_recursive(function):
    """No-op wrapper for functions with overridable recursive calls.

    :arg function: a function with parameters (value, rec), where
                   ``rec`` is expected to be a function used for
                   recursive calls.
    :returns: a function with working recursion and nothing fancy
    """
    def recursive(node):
        return function(node, recursive)
    return recursive


def noop_recursive_arg(function):
    """No-op wrapper for functions with overridable recursive calls
    and an argument.

    :arg function: a function with parameters (value, rec, arg), where
                   ``rec`` is expected to be a function used for
                   recursive calls.
    :returns: a function with working recursion and nothing fancy
    """
    def recursive(node, arg):
        return function(node, recursive, arg)
    return recursive


class Memoizer(object):
    """Caching wrapper for functions with overridable recursive calls.
    The lifetime of the cache is the lifetime of the object instance.

    :arg function: a function with parameters (value, rec), where
                   ``rec`` is expected to be a function used for
                   recursive calls.
    :returns: a function with working recursion and caching
    """
    def __init__(self, function):
        self.cache = {}
        self.function = function

    def __call__(self, node):
        try:
            return self.cache[node]
        except KeyError:
            result = self.function(node, self)
            self.cache[node] = result
            return result


class MemoizerArg(object):
    """Caching wrapper for functions with overridable recursive calls
    and an argument.  The lifetime of the cache is the lifetime of the
    object instance.

    :arg function: a function with parameters (value, rec, arg), where
                   ``rec`` is expected to be a function used for
                   recursive calls.
    :returns: a function with working recursion and caching
    """
    def __init__(self, function):
        self.cache = {}
        self.function = function

    def __call__(self, node, arg):
        cache_key = (node, arg)
        try:
            return self.cache[cache_key]
        except KeyError:
            result = self.function(node, self, arg)
            self.cache[cache_key] = result
            return result


def reuse_if_untouched(node, self):
    """Reuse if untouched recipe"""
    new_children = list(map(self, node.children))
    if all(nc == c for nc, c in zip(new_children, node.children)):
        return node
    else:
        return node.reconstruct(*new_children)


def reuse_if_untouched_arg(node, self, arg):
    """Reuse if touched recipe propagating an extra argument"""
    new_children = [self(child, arg) for child in node.children]
    if all(nc == c for nc, c in zip(new_children, node.children)):
        return node
    else:
        return node.reconstruct(*new_children)
