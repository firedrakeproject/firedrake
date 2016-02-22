"""Schedules operations to evaluate a multi-root expression DAG,
forming an ordered list of Impero terminals."""

from __future__ import absolute_import

import collections
import functools

from tsfc import gem, impero
from tsfc.node import collect_refcount


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


class ReferenceStager(object):
    """Provides staging for nodes in reference counted expression
    DAGs.  A callback function is called once the reference count is
    exhausted."""

    def __init__(self, reference_count, callback):
        """Initialises a ReferenceStager.

        :arg reference_count: initial reference counts for all
                              expected nodes
        :arg callback: function to call on each node when
                       reference count is exhausted
        """
        self.waiting = reference_count.copy()
        self.callback = callback

    def decref(self, o):
        """Decreases the reference count of a node, and possibly
        triggering a callback (when the reference count drops to
        zero)."""
        assert 1 <= self.waiting[o]

        self.waiting[o] -= 1
        if self.waiting[o] == 0:
            self.callback(o)

    def empty(self):
        """All reference counts exhausted?"""
        return not any(self.waiting.values())


class Queue(object):
    """Special queue for operation scheduling.  GEM / Impero nodes are
    inserted when they are ready to be scheduled, i.e. any operation
    which depends on the operation to be inserted must have been
    scheduled already.  This class implements a heuristic for ordering
    operations within the constraints in a way which aims to achieve
    maximum loop fusion to minimise the size of temporaries which need
    to be introduced.
    """
    def __init__(self, callback):
        """Initialises a Queue.

        :arg callback: function called on each element "popped" from the queue
        """
        # Must have deterministic iteration over the queue
        self.queue = OrderedDefaultDict(list)
        self.callback = callback

    def insert(self, indices, elem):
        """Insert element into queue.

        :arg indices: loop indices used by the scheduling heuristic
        :arg elem: element to be scheduled
        """
        self.queue[indices].append(elem)

    def process(self):
        """Pops elements from the queue and calls the callback
        function on them until the queue is empty.  The callback
        function can insert further elements into the queue.
        """
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
                self.callback(self.queue[indices].pop())
            del self.queue[indices]


def handle(ops, push, decref, node):
    """Helper function for scheduling"""
    if isinstance(node, gem.Variable):
        # Declared in the kernel header
        pass
    elif isinstance(node, gem.Literal):
        # Constant literals inlined, unless tensor-valued
        if node.shape:
            ops.append(impero.Evaluate(node))
    elif isinstance(node, gem.Zero):  # should rarely happen
        assert not node.shape
    elif isinstance(node, gem.Indexed):
        # Indexing always inlined
        decref(node.children[0])
    elif isinstance(node, gem.IndexSum):
        push(impero.Accumulate(node))
    elif isinstance(node, gem.Node):
        ops.append(impero.Evaluate(node))
        for child in node.children:
            decref(child)
    elif isinstance(node, impero.Initialise):
        ops.append(node)
    elif isinstance(node, impero.Accumulate):
        ops.append(node)
        push(impero.Initialise(node.indexsum))
        decref(node.indexsum.children[0])
    elif isinstance(node, impero.Return):
        ops.append(node)
        decref(node.expression)
    elif isinstance(node, impero.ReturnAccumulate):
        ops.append(node)
        decref(node.indexsum.children[0])
    else:
        raise AssertionError("no handler for node type %s" % type(node))


def emit_operations(assignments, get_indices):
    """Makes an ordering of operations to evaluate a multi-root
    expression DAG.

    :arg assignments: Iterable of (variable, expression) pairs.
                      The value of expression is written into variable
                      upon execution.
    :arg get_indices: mapping from GEM nodes to an ordering of free
                      indices
    :returns: list of Impero terminals correctly ordered to evaluate
              the assignments
    """
    # Filter out zeros
    assignments = [(variable, expression)
                   for variable, expression in assignments
                   if not isinstance(expression, gem.Zero)]

    # Prepare reference counts
    refcount = collect_refcount([e for v, e in assignments])

    # Stage return operations
    staging = []
    for variable, expression in assignments:
        if refcount[expression] == 1 and isinstance(expression, gem.IndexSum) \
                and set(variable.free_indices) == set(expression.free_indices):
            staging.append(impero.ReturnAccumulate(variable, expression))
            refcount[expression] -= 1
        else:
            staging.append(impero.Return(variable, expression))

    # Prepare data structures
    def push_node(node):
        queue.insert(get_indices(node), node)

    def push_op(op):
        queue.insert(op.loop_shape(get_indices), op)

    ops = []

    stager = ReferenceStager(refcount, push_node)
    queue = Queue(functools.partial(handle, ops, push_op, stager.decref))

    # Enqueue return operations
    for op in staging:
        push_op(op)

    # Schedule operations
    queue.process()

    # Assert that nothing left unprocessed
    assert stager.empty()

    # Return
    ops.reverse()
    return ops
