from __future__ import absolute_import

from singledispatch import singledispatch

from tsfc.node import Memoizer, MemoizerArg, reuse_if_untouched, reuse_if_untouched_arg
from tsfc.gem import Node, Zero, Sum, Indexed, IndexSum, ComponentTensor


@singledispatch
def replace_indices(node, self, subst):
    """Replace free indices in a GEM expression.

    :arg node: root of the expression
    :arg self: function for recursive calls
    :arg subst: tuple of pairs; each pair is a substitution
                rule with a free index to replace and an index to
                replace with.
    """
    raise AssertionError("cannot handle type %s" % type(node))


replace_indices.register(Node)(reuse_if_untouched_arg)


@replace_indices.register(Indexed)  # noqa
def _(node, self, subst):
    child, = node.children
    substitute = dict(subst)
    multiindex = tuple(substitute.get(i, i) for i in node.multiindex)
    if isinstance(child, ComponentTensor):
        # Indexing into ComponentTensor
        # Inline ComponentTensor and augment the substitution rules
        substitute.update(zip(child.multiindex, multiindex))
        return self(child.children[0], tuple(sorted(substitute.items())))
    else:
        # Replace indices
        new_child = self(child, subst)
        if new_child == child and multiindex == node.multiindex:
            return node
        else:
            return Indexed(new_child, multiindex)


def filtered_replace_indices(node, self, subst):
    """Wrapper for :func:`replace_indices`.  At each call removes
    substitution rules that do not apply."""
    filtered_subst = tuple((k, v) for k, v in subst if k in node.free_indices)
    return replace_indices(node, self, filtered_subst)


def filtered_replace_indices_top(node, self, subst):
    """Wrapper for :func:`replace_indices`.  At each call removes
    substitution rules that do not apply.  Stops recursion when there
    is nothing to substitute."""
    filtered_subst = tuple((k, v) for k, v in subst if k in node.free_indices)
    if filtered_subst:
        return replace_indices(node, self, filtered_subst)
    else:
        return node


def remove_componenttensors(expressions):
    """Removes all ComponentTensors from a list of expression DAGs."""
    mapper = MemoizerArg(filtered_replace_indices)
    return [mapper(expression, ()) for expression in expressions]


@singledispatch
def _unroll_indexsum(node, self):
    """Unrolls IndexSums below a certain extent.

    :arg node: root of the expression
    :arg self: function for recursive calls
    """
    raise AssertionError("cannot handle type %s" % type(node))


_unroll_indexsum.register(Node)(reuse_if_untouched)


@_unroll_indexsum.register(IndexSum)  # noqa
def _(node, self):
    if node.index.extent <= self.max_extent:
        # Unrolling
        summand = self(node.children[0])
        return reduce(Sum,
                      (self.replace(summand, ((node.index, i),))
                       for i in range(node.index.extent)),
                      Zero())
    else:
        return reuse_if_untouched(node, self)


def unroll_indexsum(expressions, max_extent):
    """Unrolls IndexSums below a specified extent.

    :arg expressions: list of expression DAGs
    :arg max_extent: maximum extent for which IndexSums are unrolled
    :returns: list of expression DAGs with some unrolled IndexSums
    """
    mapper = Memoizer(_unroll_indexsum)
    mapper.max_extent = max_extent
    mapper.replace = MemoizerArg(filtered_replace_indices_top)
    return map(mapper, expressions)
