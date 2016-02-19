"""
An interpreter for GEM trees.
"""
from __future__ import absolute_import

import numpy
import operator
import math
from singledispatch import singledispatch
import itertools

from tsfc import gem, node

__all__ = ("evaluate", )


class Result(object):
    """An array object that tracks which axes of the array correspond to
    gem free indices (and what those free indices are).

    :arg arr: The array.
    :arg fids: The free indices.

    The first ``len(fids)`` axes of the provided array correspond to
    the free indices, the remaining axes are the shape of each entry.
    """
    def __init__(self, arr, fids=None):
        self.arr = arr
        self.fids = fids if fids is not None else ()

    def filter(self, idx, fids):
        """Given an index tuple and some free indices, return a
        "filtered" index tuple which removes entries that correspond
        to indices in fids that are not in ``self.fids``.

        :arg idx: The index tuple to filter.
        :arg fids: The free indices for the index tuple.
        """
        return tuple(idx[fids.index(i)] for i in self.fids) + idx[len(fids):]

    def __getitem__(self, idx):
        return self.arr[idx]

    def __setitem__(self, idx, val):
        self.arr[idx] = val

    @property
    def tshape(self):
        """The total shape of the result array."""
        return self.arr.shape

    @property
    def fshape(self):
        """The shape of the free index part of the result array."""
        return self.tshape[:len(self.fids)]

    @property
    def shape(self):
        """The shape of the shape part of the result array."""
        return self.tshape[len(self.fids):]

    def __repr__(self):
        return "Result(%r, %r)" % (self.arr, self.fids)

    def __str__(self):
        return repr(self)

    @classmethod
    def empty(cls, *children, **kwargs):
        """Build an empty Result object.

        :arg children: The children used to determine the shape and
            free indices.
        :kwarg dtype: The data type of the result array.
        """
        dtype = kwargs.get("dtype", float)
        assert all(children[0].shape == c.shape for c in children)
        fids = []
        for f in itertools.chain(*(c.fids for c in children)):
            if f not in fids:
                fids.append(f)
        shape = tuple(i.extent for i in fids) + children[0].shape
        return cls(numpy.empty(shape, dtype=dtype), tuple(fids))


@singledispatch
def _evaluate(expression, self):
    """Evaluate an expression using a provided callback handler.

    :arg expression: The expression to evaluation.
    :arg self: The callback handler (should provide bindings).
    """
    raise ValueError("Unhandled node type %s" % type(expression))


@_evaluate.register(gem.Zero)  # noqa: not actually redefinition
def _(e, self):
    """Zeros produce an array of zeros."""
    return Result(numpy.zeros(e.shape, dtype=float))


@_evaluate.register(gem.Literal)  # noqa: not actually redefinition
def _(e, self):
    """Literals return their array."""
    return Result(e.array)


@_evaluate.register(gem.Variable)  # noqa: not actually redefinition
def _(e, self):
    """Look up variables in the provided bindings."""
    try:
        val = self.bindings[e]
    except KeyError:
        raise ValueError("Binding for %s not found" % e)
    if val.shape != e.shape:
        raise ValueError("Binding for %s has wrong shape.  %s, not %s." %
                         (e, val.shape, e.shape))
    return Result(val)


@_evaluate.register(gem.Power)  # noqa: not actually redefinition
@_evaluate.register(gem.Division)
@_evaluate.register(gem.Product)
@_evaluate.register(gem.Sum)
def _(e, self):
    op = {gem.Product: operator.mul,
          gem.Division: operator.div,
          gem.Sum: operator.add,
          gem.Power: operator.pow}[type(e)]

    a, b = [self(o) for o in e.children]
    result = Result.empty(a, b)
    fids = result.fids
    for idx in numpy.ndindex(result.tshape):
        result[idx] = op(a[a.filter(idx, fids)], b[b.filter(idx, fids)])
    return result


@_evaluate.register(gem.MathFunction)  # noqa: not actually redefinition
def _(e, self):
    ops = [self(o) for o in e.children]
    result = Result.empty(*ops)
    names = {"abs": abs,
             "log": math.log}
    op = names[e.name]
    for idx in numpy.ndindex(result.tshape):
        result[idx] = op(*(o[o.filter(idx, result.fids)] for o in ops))
    return result


@_evaluate.register(gem.MaxValue)  # noqa: not actually redefinition
@_evaluate.register(gem.MinValue)
def _(e, self):
    ops = [self(o) for o in e.children]
    result = Result.empty(*ops)
    op = {gem.MinValue: min,
          gem.MaxValue: max}[type(e)]
    for idx in numpy.ndindex(result.tshape):
        result[idx] = op(*(o[o.filter(idx, result.fids)] for o in ops))
    return result


@_evaluate.register(gem.Comparison)  # noqa: not actually redefinition
def _(e, self):
    ops = [self(o) for o in e.children]
    op = {">": operator.gt,
          ">=": operator.ge,
          "==": operator.eq,
          "!=": operator.ne,
          "<": operator.lt,
          "<=": operator.le}[e.operator]
    result = Result.empty(*ops, dtype=bool)
    for idx in numpy.ndindex(result.tshape):
        result[idx] = op(*(o[o.filter(idx, result.fids)] for o in ops))
    return result


@_evaluate.register(gem.LogicalNot)  # noqa: not actually redefinition
def _(e, self):
    val = self(e.children[0])
    assert val.arr.dtype == numpy.dtype("bool")
    result = Result.empty(val, bool)
    for idx in numpy.ndindex(result.tshape):
        result[idx] = not val[val.filter(idx, result.fids)]
    return result


@_evaluate.register(gem.LogicalAnd)  # noqa: not actually redefinition
def _(e, self):
    a, b = [self(o) for o in e.children]
    assert a.arr.dtype == numpy.dtype("bool")
    assert b.arr.dtype == numpy.dtype("bool")
    result = Result.empty(a, b, bool)
    for idx in numpy.ndindex(result.tshape):
        result[idx] = a[a.filter(idx, result.fids)] and \
            b[b.filter(idx, result.fids)]
    return result


@_evaluate.register(gem.LogicalOr)  # noqa: not actually redefinition
def _(e, self):
    a, b = [self(o) for o in e.children]
    assert a.arr.dtype == numpy.dtype("bool")
    assert b.arr.dtype == numpy.dtype("bool")
    result = Result.empty(a, b, dtype=bool)
    for idx in numpy.ndindex(result.tshape):
        result[idx] = a[a.filter(idx, result.fids)] or \
            b[b.filter(idx, result.fids)]
    return result


@_evaluate.register(gem.Conditional)  # noqa: not actually redefinition
def _(e, self):
    cond, then, else_ = [self(o) for o in e.children]
    assert cond.arr.dtype == numpy.dtype("bool")
    result = Result.empty(cond, then, else_)
    for idx in numpy.ndindex(result.tshape):
        if cond[cond.filter(idx, result.fids)]:
            result[idx] = then[then.filter(idx, result.fids)]
        else:
            result[idx] = else_[else_.filter(idx, result.fids)]
    return result


@_evaluate.register(gem.Indexed)  # noqa: not actually redefinition
def _(e, self):
    """Indexing maps shape to free indices"""
    val = self(e.children[0])
    fids = tuple(i for i in e.multiindex if isinstance(i, gem.Index))

    idx = []
    # First pick up all the existing free indices
    for _ in val.fids:
        idx.append(Ellipsis)
    # Now grab the shape axes
    for i in e.multiindex:
        if isinstance(i, gem.Index):
            # Free index, want entire extent
            idx.append(Ellipsis)
        elif isinstance(i, gem.VariableIndex):
            # Variable index, evaluate inner expression
            result, = self(i.expression)
            assert not result.tshape
            idx.append(result[()])
        else:
            # Fixed index, just pick that value
            idx.append(i)
    assert len(idx) == len(val.tshape)
    return Result(val[idx], val.fids + fids)


@_evaluate.register(gem.ComponentTensor)  # noqa: not actually redefinition
def _(e, self):
    """Component tensors map free indices to shape."""
    val = self(e.children[0])
    axes = []
    fids = []
    # First grab the free indices that aren't bound
    for a, f in enumerate(val.fids):
        if f not in e.multiindex:
            axes.append(a)
            fids.append(f)
    # Now the bound free indices
    for i in e.multiindex:
        axes.append(val.fids.index(i))
    # Now the existing shape
    axes.extend(range(len(val.fshape), len(val.tshape)))
    return Result(numpy.transpose(val.arr, axes=axes),
                  tuple(fids))


@_evaluate.register(gem.IndexSum)  # noqa: not actually redefinition
def _(e, self):
    """Index sums reduce over the given axis."""
    val = self(e.children[0])
    idx = val.fids.index(e.index)
    return Result(val.arr.sum(axis=idx),
                  val.fids[:idx] + val.fids[idx+1:])


@_evaluate.register(gem.ListTensor)  # noqa: not actually redefinition
def _(e, self):
    """List tensors just turn into arrays."""
    ops = [self(o) for o in e.children]
    assert all(ops[0].fids == o.fids for o in ops)
    return Result(numpy.asarray([o.arr for o in ops]).reshape(e.shape),
                  ops[0].fids)


def evaluate(expressions, bindings=None):
    """Evaluate some GEM expressions given variable bindings.

    :arg expressions: A single GEM expression, or iterable of
        expressions to evaluate.
    :kwarg bindings: An optional dict mapping GEM :class:`gem.Variable`
        nodes to data.
    :returns: a list of the evaluated expressions.
    """
    try:
        exprs = tuple(expressions)
    except TypeError:
        exprs = (expressions, )
    mapper = node.Memoizer(_evaluate)
    mapper.bindings = bindings if bindings is not None else {}
    return map(mapper, exprs)
