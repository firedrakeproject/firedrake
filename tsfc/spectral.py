from collections import OrderedDict, defaultdict, namedtuple
from functools import partial, singledispatch
from itertools import chain, zip_longest

import numpy

from gem.gem import (ComponentTensor, Delta, Index, Indexed, IndexSum,
                     Literal, Node, Sum, index_sum, one)
from gem.node import Memoizer, MemoizerArg, reuse_if_untouched, traversal
from gem.optimise import filtered_replace_indices
from gem.optimise import delta_elimination as _delta_elimination
from gem.optimise import replace_division, unroll_indexsum
from gem.refactorise import ATOMIC, COMPOUND, OTHER, MonomialSum, collect_monomials
from gem.unconcatenate import unconcatenate
from gem.coffee import optimise_monomial_sum
from gem.utils import groupby


Integral = namedtuple('Integral', ['expression',
                                   'quadrature_multiindex',
                                   'argument_indices'])


# -- Exposing collocation structure to sum factorisation ---------------------
#
# On tensor-product cells with a collocated quadrature rule (e.g. a
# ``variant="spectral"`` element integrated at its own Gauss-Lobatto-Legendre
# nodes), the value tabulation is the identity.  However FInAT/TSFC materialise
# the tensor-product tabulation as a dense multi-dimensional ``Literal`` that
# factors as ``T[i, q_own, q_a, q_b] = factor[i, q_own] * const`` -- i.e. a
# genuine 1D factor spuriously *broadcast* (constant) over the other quadrature
# directions.  That broadcast hides both the low-rank structure and the
# collocation identity from sum factorisation/delta elimination, so the
# generated operator application scales like O(p^{2d}) instead of O(p^{d+1})
# (the matvec/action of the 3D high order Laplacian was ~5x slower than it
# should be as a result).
#
# The two passes below recover the structure with purely local, exact GEM
# rewrites applied before sum factorisation:
#   * ``drop_constant_literal_axes`` removes running indices on axes along which
#     a tabulation literal is constant (the broadcast), uncovering the 1D
#     factors;
#   * ``convert_identity_literals`` rewrites a resulting identity tabulation as a
#     Kronecker ``Delta`` so the delta elimination cancels the redundant
#     interpolation contraction.


def _is_identity_table(array, epsilon):
    """True if ``array`` is (numerically) a square identity matrix."""
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        return False
    return numpy.allclose(array, numpy.eye(array.shape[0], dtype=array.dtype),
                          rtol=0.0, atol=epsilon)


def _constant_axes(array, epsilon):
    """Axes of ``array`` (length > 1) along which it is constant."""
    if array.ndim < 2:
        return ()
    eps = epsilon * (1.0 + (numpy.abs(array).max() if array.size else 0.0))
    axes = []
    for axis in range(array.ndim):
        if array.shape[axis] <= 1:
            continue
        spread = numpy.ptp(array, axis=axis)
        if (spread.max() if spread.size else 0.0) <= eps:
            axes.append(axis)
    return tuple(axes)


def _anchored_indices(expressions, epsilon):
    """Indices that are safe to drop from a constant literal axis.

    Dropping a running index from a tabulation literal removes that index from
    the expression.  This is only sound if the index is *anchored*: it must also
    occur somewhere other than a constant axis of an ``Indexed(Literal(...))``,
    so that it remains present afterwards (otherwise an enclosing
    ``ComponentTensor``/``IndexSum`` that binds or sums it would be left
    referencing a vanished index).

    An index is anchored iff it is *introduced* by some node other than via a
    constant literal axis.  The indices a node introduces are exactly its free
    indices minus those of its children; for ``Indexed(Literal(...))`` the
    constant axes are excluded.

    Indices bound by a ``ComponentTensor``/``IndexSum`` anywhere in the DAG are
    never anchored.  The anchoring analysis is global, but binding is scoped and
    GEM is a shared DAG, so an index can occur non-constantly under one binder
    yet appear *only* on a constant literal axis within the scope of another
    binder of the same ``Index`` object.  Dropping it there would orphan that
    binder's multiindex; refusing to drop any bound index avoids this while
    still exposing the (free) broadcast quadrature directions we target.
    """
    anchored = set()
    bound = set()
    for node in traversal(expressions):
        if isinstance(node, (ComponentTensor, IndexSum)):
            bound.update(node.multiindex)
        child_free = set()
        for child in node.children:
            child_free |= set(child.free_indices)
        own = set(node.free_indices) - child_free
        if isinstance(node, Indexed) and isinstance(node.children[0], Literal):
            const = set(_constant_axes(node.children[0].array, epsilon))
            own = {index for axis, index in enumerate(node.multiindex)
                   if isinstance(index, Index) and axis not in const}
        anchored |= own
    return anchored - bound


@singledispatch
def _drop_constant_axes(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_drop_constant_axes.register(Node)(reuse_if_untouched)


@_drop_constant_axes.register(Indexed)
def _drop_constant_axes_indexed(node, self):
    child, = node.children
    # If the literal genuinely does not vary along an axis indexed by a running
    # index, indexing it there is redundant: drop the axis (its value is any
    # slice).  Only drop *anchored* indices (those that also occur elsewhere), so
    # the dropped index remains present in the expression and any enclosing
    # ComponentTensor/IndexSum that references it stays well formed.
    if (isinstance(child, Literal) and len(node.multiindex) == child.array.ndim
            and child.array.ndim >= 2):
        const = set(_constant_axes(child.array, self.epsilon))
        slicer = []
        new_multiindex = []
        dropped = False
        for axis, index in enumerate(node.multiindex):
            if (axis in const and isinstance(index, Index)
                    and index in self.anchored):
                slicer.append(0)  # constant along this axis: keep slice 0
                dropped = True
            else:
                slicer.append(slice(None))
                new_multiindex.append(index)
        if dropped:
            reduced = child.array[tuple(slicer)]
            return Indexed(Literal(reduced, dtype=child.dtype),
                           tuple(new_multiindex))
    return reuse_if_untouched(node, self)


def drop_constant_literal_axes(expressions, epsilon=1e-12):
    """Drop running indices of ``Indexed(Literal(...))`` along axes on which the
    literal is constant, exposing the underlying low-rank tabulation structure
    to sum factorisation.

    Only indices that are anchored elsewhere (see :func:`_anchored_indices`) are
    dropped, so the rewrite never leaves a dangling index behind.

    :arg expressions: iterable of GEM expressions
    :arg epsilon: tolerance for recognising a constant axis
    """
    expressions = list(expressions)
    mapper = Memoizer(_drop_constant_axes)
    mapper.epsilon = epsilon
    mapper.anchored = _anchored_indices(expressions, epsilon)
    return [mapper(e) for e in expressions]


@singledispatch
def _identity_to_delta(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_identity_to_delta.register(Node)(reuse_if_untouched)


@_identity_to_delta.register(Indexed)
def _identity_to_delta_indexed(node, self):
    child, = node.children
    # A collocated tabulation matrix (CG nodal values at the collocated
    # quadrature points) is the identity.  Indexing it with two distinct running
    # indices is exactly a Kronecker delta; rewriting it as such lets sum
    # factorisation/delta elimination cancel the redundant contraction.
    if (isinstance(child, Literal) and len(node.multiindex) == 2
            and all(isinstance(i, Index) for i in node.multiindex)
            and _is_identity_table(child.array, self.epsilon)):
        i, j = node.multiindex
        return Delta(i, j)
    return reuse_if_untouched(node, self)


def convert_identity_literals(expressions, epsilon=1e-12):
    """Rewrite ``Indexed(Literal(I), (i, j))`` as ``Delta(i, j)`` for identity
    tabulation matrices, exposing collocation structure to sum factorisation.

    :arg expressions: iterable of GEM expressions
    :arg epsilon: tolerance for recognising an identity matrix
    """
    mapper = Memoizer(_identity_to_delta)
    mapper.epsilon = epsilon
    return [mapper(e) for e in expressions]


def Integrals(expressions, quadrature_multiindex, argument_multiindices, parameters):
    """Constructs an integral representation for each GEM integrand
    expression.

    :arg expressions: integrand multiplied with quadrature weight;
                      multi-root GEM expression DAG
    :arg quadrature_multiindex: quadrature multiindex (tuple)
    :arg argument_multiindices: tuple of argument multiindices,
                                one multiindex for each argument
    :arg parameters: parameters dictionary

    :returns: list of integral representations
    """
    # Rewrite: a / b => a * (1 / b)
    expressions = replace_division(expressions)

    # Expose the sum-factorisable structure of tensor-product tabulation
    # matrices.  First drop running indices on axes where a tabulation literal
    # is merely a constant broadcast (a spurious coupling to the other
    # quadrature directions); this uncovers the genuine 1D factors.  Then
    # rewrite any resulting identity tabulation (collocated nodal values) as a
    # Kronecker delta so the delta elimination below cancels the redundant
    # interpolation contraction.  Together these turn the O(p^{2d}) collocated
    # operator application into the O(p^{d+1}) sum-factorised form.
    expressions = drop_constant_literal_axes(expressions)
    expressions = convert_identity_literals(expressions)

    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        expressions = unroll_indexsum(expressions, predicate=predicate)

    expressions = [index_sum(e, quadrature_multiindex) for e in expressions]
    argument_indices = tuple(chain.from_iterable(argument_multiindices))
    return [Integral(e, quadrature_multiindex, argument_indices) for e in expressions]


def _delta_inside(node, self):
    """Does node contain a Delta?"""
    return any(isinstance(child, Delta) or self(child)
               for child in node.children)


def flatten(var_reps, index_cache):
    quadrature_indices = OrderedDict()

    pairs = []  # assignment pairs
    for variable, reps in var_reps:
        # Extract argument indices
        argument_indices, = set(r.argument_indices for r in reps)
        assert set(variable.free_indices) == set(argument_indices)

        # Extract and verify expressions
        expressions = [r.expression for r in reps]
        assert all(set(e.free_indices) <= set(argument_indices)
                   for e in expressions)

        # Save assignment pair
        pairs.append((variable, Sum(*expressions)))

        # Collect quadrature_indices
        for r in reps:
            quadrature_indices.update(zip_longest(r.quadrature_multiindex, ()))

    # Split Concatenate nodes
    pairs = unconcatenate(pairs, cache=index_cache)

    def group_key(pair):
        variable, expression = pair
        return frozenset(variable.free_indices)

    # Common memoizer to remove ComponentTensors
    index_replacer = MemoizerArg(filtered_replace_indices)
    # Common memoizer to test for Deltas inside expressions
    delta_inside = Memoizer(_delta_inside)
    # Variable ordering after delta cancellation
    narrow_variables = OrderedDict()
    # Assignments are variable -> MonomialSum map
    delta_simplified = defaultdict(MonomialSum)
    # Group assignment pairs by argument indices
    for free_indices, pair_group in groupby(pairs, group_key):
        variables, expressions = zip(*pair_group)
        # Argument factorise expressions
        classifier = partial(classify, set(free_indices), delta_inside=delta_inside)
        monomial_sums = collect_monomials(expressions, classifier)
        # For each monomial, apply delta cancellation and insert
        # result into delta_simplified.
        for variable, monomial_sum in zip(variables, monomial_sums):
            for monomial in monomial_sum:
                var, s, a, r = delta_elimination(variable, *monomial, index_replacer)
                narrow_variables.setdefault(var)
                delta_simplified[var].add(s, a, r)

    # Final factorisation
    for variable in narrow_variables:
        monomial_sum = delta_simplified[variable]
        # Collect sum indices applicable to the current MonomialSum
        sum_indices = set(chain.from_iterable(m.sum_indices for m in monomial_sum))
        # Put them in a deterministic order
        sum_indices = [i for i in quadrature_indices if i in sum_indices]
        # Sort for increasing index extent, this obtains the good
        # factorisation for triangle x interval cells.  Python sort is
        # stable, so in the common case when index extents are equal,
        # the previous deterministic ordering applies which is good
        # for getting smaller temporaries.
        sum_indices = sorted(sum_indices, key=lambda index: index.extent)
        # Apply sum factorisation combined with COFFEE technology
        expression = sum_factorise(variable, sum_indices, monomial_sum)
        yield (variable, expression)


# Lower any Deltas that survive sum factorisation (e.g. test-function
# collocation deltas introduced by convert_identity_literals that could not be
# cancelled against a sum index) to identity indexing for code generation.
finalise_options = dict(replace_delta=True)


def classify(argument_indices, expression, delta_inside):
    """Classifier for argument factorisation"""
    n = len(argument_indices.intersection(expression.free_indices))
    if n == 0:
        return OTHER
    elif n == 1:
        if isinstance(expression, (Delta, Indexed)) and not delta_inside(expression):
            return ATOMIC
        else:
            return COMPOUND
    else:
        return COMPOUND


def delta_elimination(variable, sum_indices, args, rest, index_replacer):
    """IndexSum-Delta cancellation for monomials."""
    factors = list(args) + [variable, rest]  # construct factors

    def prune(factors):
        # Skip last factor (``rest``, see above) which can be
        # arbitrarily complicated, so its pruning may be expensive,
        # and its early pruning brings no advantages.
        result = [index_replacer(f, ()) for f in factors[:-1]]
        result.append(factors[-1])
        return result

    # Cancel sum indices
    sum_indices, factors = _delta_elimination(sum_indices, factors)
    factors = prune(factors)

    # Cancel variable indices
    var_indices, factors = _delta_elimination(variable.free_indices, factors)
    factors = prune(factors)

    # Destructure factors after cancellation
    rest = factors.pop()
    variable = factors.pop()
    args = [f for f in factors if f != one]

    assert set(var_indices) == set(variable.free_indices)
    return variable, sum_indices, args, rest


def sum_factorise(variable, tail_ordering, monomial_sum):
    if tail_ordering:
        key_ordering = OrderedDict()
        sub_monosums = defaultdict(MonomialSum)
        for sum_indices, atomics, rest in monomial_sum:
            # Pull out those sum indices that are not contained in the
            # tail ordering, together with those atomics which do not
            # share free indices with the tail ordering.
            #
            # Based on this, split the monomial sum, then recursively
            # optimise each sub monomial sum with the first tail index
            # removed.
            tail_indices = tuple(i for i in sum_indices if i in tail_ordering)
            tail_atomics = tuple(a for a in atomics
                                 if set(tail_indices) & set(a.free_indices))
            head_indices = tuple(i for i in sum_indices if i not in tail_ordering)
            head_atomics = tuple(a for a in atomics if a not in tail_atomics)
            key = (head_indices, head_atomics)
            key_ordering.setdefault(key)
            sub_monosums[key].add(tail_indices, tail_atomics, rest)
        sub_monosums = [(k, sub_monosums[k]) for k in key_ordering]

        monomial_sum = MonomialSum()
        for (sum_indices, atomics), monosum in sub_monosums:
            new_rest = sum_factorise(variable, tail_ordering[1:], monosum)
            monomial_sum.add(sum_indices, atomics, new_rest)

    # Use COFFEE algorithm to optimise the monomial sum
    return optimise_monomial_sum(monomial_sum, variable.index_ordering())
