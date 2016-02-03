"""Generic abstract node class and utility functions for creating
expression DAG languages."""

from __future__ import absolute_import


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

    def __getinitargs__(self, children):
        """Constructs an argument list for the constructor with
        non-child data from 'self' and children from 'children'.

        Internally used utility function.
        """
        front_args = [getattr(self, name) for name in self.__front__]
        back_args = [getattr(self, name) for name in self.__back__]

        return tuple(front_args) + tuple(children) + tuple(back_args)

    def reconstruct(self, *args):
        """Reconstructs the node with new children from
        'args'. Non-child data are copied from 'self'.

        Returns a new object.
        """
        return type(self)(*self.__getinitargs__(args))

    def __repr__(self):
        init_args = self.__getinitargs__(self.children)
        return "%s(%s)" % (type(self).__name__, ", ".join(map(repr, init_args)))

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
        self_initargs = self.__getinitargs__(self.children)
        other_initargs = other.__getinitargs__(other.children)
        return self_initargs == other_initargs

    def get_hash(self):
        """Hash function.

        This is the method to potentially override in derived classes,
        not :meth:`__hash__`.
        """
        return hash((type(self),) + self.__getinitargs__(self.children))


def traversal(expression_dags):
    """Pre-order traversal of the nodes of expression DAGs."""
    seen = set(expression_dags)
    lifo = list(expression_dags)

    while lifo:
        node = lifo.pop()
        yield node
        for child in node.children:
            if child not in seen:
                seen.add(child)
                lifo.append(child)
