from __future__ import absolute_import

import collections

from singledispatch import singledispatch

from tsfc import gem as ein, impero as imp
from tsfc.node import traversal


class OrderedDefaultDict(collections.OrderedDict):
    """A dictionary that provides a default value and ordered iteration.

    :arg factory: The callable used to create the default value.

    See :class:`collections.OrderedDict` for description of the
    remaining arguments.
    """
    def __init__(self, factory, *args, **kwargs):
        self.factory = factory
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        val = self[key] = self.factory()
        return val


class Queue(object):
    def __init__(self, reference_count, get_indices):
        self.waiting = reference_count.copy()
        # Need to have deterministic iteration over the queue.
        self.queue = OrderedDefaultDict(list)
        self.get_indices = get_indices

    def reference(self, o):
        if o not in self.waiting:
            return

        assert 1 <= self.waiting[o]

        self.waiting[o] -= 1
        if self.waiting[o] == 0:
            self.insert(o, self.get_indices(o))

    def insert(self, o, indices):
        self.queue[indices].append(o)

    def __iter__(self):
        indices = ()
        while self.queue:
            # Find innermost non-empty outer loop
            while indices not in (i[:len(indices)] for i in self.queue.keys()):
                indices = indices[:-1]

            # Pick a loop
            for i in self.queue.keys():
                if i[:len(indices)] == indices:
                    indices = i
                    break

            while self.queue[indices]:
                yield self.queue[indices].pop()
            del self.queue[indices]


def count_references(expressions):
    result = collections.Counter(expressions)
    for node in traversal(expressions):
        result.update(node.children)
    return result


@singledispatch
def impero_indices(node, indices):
    raise AssertionError("Cannot handle type: %s" % type(node))


@impero_indices.register(imp.Return)  # noqa: Not actually redefinition
def _(node, indices):
    assert set(node.variable.free_indices) >= set(node.expression.free_indices)
    return indices(node.variable)


@impero_indices.register(imp.Initialise)  # noqa: Not actually redefinition
def _(node, indices):
    return indices(node.indexsum)


@impero_indices.register(imp.Accumulate)  # noqa: Not actually redefinition
def _(node, indices):
    return indices(node.indexsum.children[0])


@impero_indices.register(imp.Evaluate)  # noqa: Not actually redefinition
def _(node, indices):
    return indices(node.expression)


@singledispatch
def handle(node, enqueue, emit):
    raise AssertionError("Cannot handle foreign type: %s" % type(node))


@handle.register(ein.Node)  # noqa: Not actually redefinition
def _(node, enqueue, emit):
    emit(imp.Evaluate(node))
    for child in node.children:
        enqueue(child)


@handle.register(ein.Variable)  # noqa: Not actually redefinition
def _(node, enqueue, emit):
    pass


@handle.register(ein.Literal)  # noqa: Not actually redefinition
@handle.register(ein.Zero)
def _(node, enqueue, emit):
    if node.shape:
        emit(imp.Evaluate(node))


@handle.register(ein.Indexed)  # noqa: Not actually redefinition
def _(node, enqueue, emit):
    enqueue(node.children[0])


@handle.register(ein.IndexSum)  # noqa: Not actually redefinition
def _(node, enqueue, emit):
    enqueue(imp.Accumulate(node))


@handle.register(imp.Initialise)  # noqa: Not actually redefinition
def _(op, enqueue, emit):
    emit(op)


@handle.register(imp.Accumulate)  # noqa: Not actually redefinition
def _(op, enqueue, emit):
    emit(op)
    enqueue(imp.Initialise(op.indexsum))
    enqueue(op.indexsum.children[0])


@handle.register(imp.Return)  # noqa: Not actually redefinition
def _(op, enqueue, emit):
    emit(op)
    enqueue(op.expression)


def make_ordering(assignments, indices_map):
    assignments = filter(lambda x: not isinstance(x[1], ein.Zero), assignments)
    expressions = [expression for variable, expression in assignments]
    queue = Queue(count_references(expressions), indices_map)

    for variable, expression in assignments:
        queue.insert(imp.Return(variable, expression), indices_map(expression))

    result = []

    def emit(op):
        return result.append((impero_indices(op, indices_map), op))

    def enqueue(item):
        if isinstance(item, ein.Node):
            queue.reference(item)
        elif isinstance(item, imp.Node):
            queue.insert(item, impero_indices(item, indices_map))
        else:
            raise AssertionError("should never happen")

    for o in queue:
        handle(o, enqueue, emit)
    assert not any(queue.waiting.values())
    return list(reversed(result))
