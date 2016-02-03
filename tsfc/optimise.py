from __future__ import absolute_import

import collections

from singledispatch import singledispatch

from tsfc.node import traversal
from tsfc.gem import (Node, Literal, Zero, Sum, Index, Indexed,
                      IndexSum, ComponentTensor, ListTensor)


def inline_indices(expression, result_cache):
    def cached_handle(node, subst):
        cache_key = (node, tuple(sorted(subst.items())))
        try:
            return result_cache[cache_key]
        except KeyError:
            result = handle(node, subst)
            result_cache[cache_key] = result
            return result

    @singledispatch
    def handle(node, subst):
        raise AssertionError("Cannot handle foreign type: %s" % type(node))

    @handle.register(Node)  # noqa: Not actually redefinition
    def _(node, subst):
        new_children = [cached_handle(child, subst) for child in node.children]
        if all(nc == c for nc, c in zip(new_children, node.children)):
            return node
        else:
            return node.reconstruct(*new_children)

    @handle.register(Indexed)  # noqa: Not actually redefinition
    def _(node, subst):
        child, = node.children
        multiindex = tuple(subst.get(i, i) for i in node.multiindex)
        if isinstance(child, ComponentTensor):
            new_subst = dict(zip(child.multiindex, multiindex))
            composed_subst = {k: new_subst.get(v, v) for k, v in subst.items()}
            composed_subst.update(new_subst)
            filtered_subst = {k: v for k, v in composed_subst.items() if k in child.children[0].free_indices}
            return cached_handle(child.children[0], filtered_subst)
        elif isinstance(child, ListTensor) and all(isinstance(i, int) for i in multiindex):
            return cached_handle(child.array[multiindex], subst)
        elif isinstance(child, Literal) and all(isinstance(i, int) for i in multiindex):
            return Literal(child.array[multiindex])
        else:
            new_child = cached_handle(child, subst)
            if new_child == child and multiindex == node.multiindex:
                return node
            else:
                return Indexed(new_child, multiindex)

    return cached_handle(expression, {})


def expand_indexsum(expressions, max_extent):
    result_cache = {}

    def cached_handle(node):
        try:
            return result_cache[node]
        except KeyError:
            result = handle(node)
            result_cache[node] = result
            return result

    @singledispatch
    def handle(node):
        raise AssertionError("Cannot handle foreign type: %s" % type(node))

    @handle.register(Node)  # noqa: Not actually redefinition
    def _(node):
        new_children = [cached_handle(child) for child in node.children]
        if all(nc == c for nc, c in zip(new_children, node.children)):
            return node
        else:
            return node.reconstruct(*new_children)

    @handle.register(IndexSum)  # noqa: Not actually redefinition
    def _(node):
        if node.index.extent <= max_extent:
            summand = cached_handle(node.children[0])
            ct = ComponentTensor(summand, (node.index,))
            result = Zero()
            for i in xrange(node.index.extent):
                result = Sum(result, Indexed(ct, (i,)))
            return result
        else:
            return node.reconstruct(*[cached_handle(child) for child in node.children])

    return [cached_handle(expression) for expression in expressions]


def collect_index_extents(expression):
    result = collections.OrderedDict()

    for node in traversal([expression]):
        if isinstance(node, Indexed):
            assert len(node.multiindex) == len(node.children[0].shape)
            for index, extent in zip(node.multiindex, node.children[0].shape):
                if isinstance(index, Index):
                    if index not in result:
                        result[index] = extent
                    elif result[index] != extent:
                        raise AssertionError("Inconsistent index extents!")

    return result
