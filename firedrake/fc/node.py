from __future__ import absolute_import


class Node(object):
    __slots__ = ('hash_value',)

    __front__ = ()
    __back__ = ()

    def __getinitargs__(self, children):
        front_args = [getattr(self, name) for name in self.__front__]
        back_args = [getattr(self, name) for name in self.__back__]

        return tuple(front_args) + tuple(children) + tuple(back_args)

    def reconstruct(self, *args):
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
        if type(self) != type(other):
            return False
        self_initargs = self.__getinitargs__(self.children)
        other_initargs = other.__getinitargs__(other.children)
        return self_initargs == other_initargs

    def get_hash(self):
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
