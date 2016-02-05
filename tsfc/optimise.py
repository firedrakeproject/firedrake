from __future__ import absolute_import

from singledispatch import singledispatch

from tsfc.gem import (Node, Literal, Zero, Sum, Indexed,
                      IndexSum, ComponentTensor, ListTensor)


class Memoizer(object):
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


class MemoizerWithArgs(object):
    def __init__(self, function, argskeyfunc):
        self.cache = {}
        self.function = function
        self.argskeyfunc = argskeyfunc

    def __call__(self, node, *args, **kwargs):
        cache_key = (node, self.argskeyfunc(*args, **kwargs))
        try:
            return self.cache[cache_key]
        except KeyError:
            result = self.function(node, self, *args, **kwargs)
            self.cache[cache_key] = result
            return result


def reuse_if_untouched(node, self):
    new_children = map(self, node.children)
    if all(nc == c for nc, c in zip(new_children, node.children)):
        return node
    else:
        return node.reconstruct(*new_children)


def reuse_if_untouched_with_args(node, self, *args, **kwargs):
    new_children = [self(child, *args, **kwargs) for child in node.children]
    if all(nc == c for nc, c in zip(new_children, node.children)):
        return node
    else:
        return node.reconstruct(*new_children)


@singledispatch
def replace_indices(node, self, subst):
    raise AssertionError("cannot handle type %s" % type(node))


replace_indices.register(Node)(reuse_if_untouched_with_args)


@replace_indices.register(Indexed)  # noqa
def _(node, self, subst):
    child, = node.children
    multiindex = tuple(subst.get(i, i) for i in node.multiindex)
    if isinstance(child, ComponentTensor):
        new_subst = dict(zip(child.multiindex, multiindex))
        composed_subst = {k: new_subst.get(v, v) for k, v in subst.items()}
        composed_subst.update(new_subst)
        return self(child.children[0], composed_subst)
    elif isinstance(child, ListTensor) and all(isinstance(i, int) for i in multiindex):
        return self(child.array[multiindex], subst)
    elif isinstance(child, Literal) and all(isinstance(i, int) for i in multiindex):
        return Literal(child.array[multiindex])
    else:
        new_child = self(child, subst)
        if new_child == child and multiindex == node.multiindex:
            return node
        else:
            return Indexed(new_child, multiindex)


def remove_componenttensors(expressions):
    def argskeyfunc(subst):
        return tuple(sorted(subst.items()))

    def filtered(node, self, subst):
        filtered_subst = {k: v for k, v in subst.items() if k in node.free_indices}
        return replace_indices(node, self, filtered_subst)

    mapper = MemoizerWithArgs(filtered, argskeyfunc)
    return [mapper(expression, {}) for expression in expressions]


@singledispatch
def _unroll_indexsum(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_unroll_indexsum.register(Node)(reuse_if_untouched)


@_unroll_indexsum.register(IndexSum)  # noqa
def _(node, self):
    if node.index.extent <= self.max_extent:
        summand = self(node.children[0])
        ct = ComponentTensor(summand, (node.index,))
        return reduce(Sum,
                      (Indexed(ct, (i,))
                       for i in range(node.index.extent)),
                      Zero())
    else:
        return reuse_if_untouched(node, self)


def unroll_indexsum(expressions, max_extent):
    mapper = Memoizer(_unroll_indexsum)
    mapper.max_extent = max_extent
    return map(mapper, expressions)
