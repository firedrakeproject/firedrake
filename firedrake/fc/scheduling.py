from __future__ import absolute_import

import collections

from singledispatch import singledispatch

from firedrake.fc import einstein as ein, impero as imp
from firedrake.fc.node import traversal


class Queue(object):
    def __init__(self, reference_count, get_indices):
        self.waiting = reference_count.copy()
        self.queue = collections.defaultdict(list)
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


def count_references(expression):
    result = collections.Counter()
    result[expression] += 1
    for node in traversal(expression):
        for child in node.children:
            result[child] += 1
    return result


@singledispatch
def impero_indices(node, indices):
    raise AssertionError("Cannot handle type: %s" % type(node))


@impero_indices.register(imp.Return)
def _(node, indices):
    assert set(node.variable.free_indices) >= set(node.expression.free_indices)
    return indices(node.variable)


@impero_indices.register(imp.Initialise)
def _(node, indices):
    return indices(node.indexsum)


@impero_indices.register(imp.Accumulate)
def _(node, indices):
    return indices(node.indexsum.children[0])


@impero_indices.register(imp.Evaluate)
def _(node, indices):
    return indices(node.expression)


@singledispatch
def handle(node, enqueue, emit):
    raise AssertionError("Cannot handle foreign type: %s" % type(node))


@handle.register(ein.Node)
def _(node, enqueue, emit):
    emit(imp.Evaluate(node))
    for child in node.children:
        enqueue(child)


@handle.register(ein.Variable)
def _(node, enqueue, emit):
    pass


@handle.register(ein.Literal)
def _(node, enqueue, emit):
    if node.shape:
        emit(imp.Evaluate(node))


@handle.register(ein.Indexed)
def _(node, enqueue, emit):
    enqueue(node.children[0])


@handle.register(ein.IndexSum)
def _(node, enqueue, emit):
    enqueue(imp.Accumulate(node))


@handle.register(imp.Initialise)
def _(op, enqueue, emit):
    emit(op)


@handle.register(imp.Accumulate)
def _(op, enqueue, emit):
    emit(op)
    enqueue(imp.Initialise(op.indexsum))
    enqueue(op.indexsum.children[0])


@handle.register(imp.Return)
def _(op, enqueue, emit):
    emit(op)
    enqueue(op.expression)


def make_ordering(variable, expression, indices_map):
    queue = Queue(count_references(expression), indices_map)
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
    return list(reversed(result))
