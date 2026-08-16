"""Apply structure-preserving optimization to finite element integrals.

The order of transformations is part of the algorithm.  Argument
factorization first identifies finite element linear maps without expanding
their basis transformations.  Delta cancellation then exposes the legal
contractions.  Sum factorization places quadrature reductions, and COFFEE
eliminates scalar sharing at every reduction level.  In this form,
sum-factorization is generalized code motion on a spectral loop nest rather
than a separate algebraic optimization pipeline.
"""

from collections import OrderedDict, defaultdict, namedtuple
from functools import partial
from itertools import chain, permutations, zip_longest

import numpy

from gem import impero_utils
from gem.gem import Conditional, Delta, Indexed, Node, Sum, index_sum, one
from gem.contraction import estimate_cost
from gem.node import Memoizer, MemoizerArg
from gem.optimise import constant_fold_zero, filtered_replace_indices
from gem.optimise import delta_elimination as _delta_elimination
from gem.optimise import replace_division, unroll_indexsum
from gem.refactorise import ATOMIC, COMPOUND, OTHER, MonomialSum, collect_monomials
from gem.unconcatenate import unconcatenate
from gem.coffee import optimise_monomial_sum
from gem.utils import groupby


Integral = namedtuple('Integral', ['expression',
                                   'quadrature_multiindex',
                                   'argument_indices'])

Plan = tuple[tuple[Node, Node], ...]

# Cache-resident working set: 8192 doubles = 64 KiB of contraction temporaries.
storage_budget = 8192


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


def _declared_storage(plan: Plan, quadrature_indices: tuple) -> int:
    """Count the temporary entries this plan makes Impero declare.

    Only a schedule fixes how wide a temporary must be, because that width
    follows from the loop nest a value outlives, which an expression DAG
    does not record.

    Parameters
    ----------
    plan
        Output variables and their factorized GEM expressions.
    quadrature_indices
        Every quadrature index of the integral, in source order.

    Returns
    -------
    int
        Scalar entries in the temporaries, or zero if the plan schedules
        to nothing.

    """
    variables = [variable for variable, _ in plan]
    expressions = impero_utils.preprocess_gem(
        constant_fold_zero([expression for _, expression in plan]),
        **finalise_options)
    ordering = quadrature_indices + tuple(chain.from_iterable(
        variable.index_ordering() for variable in variables))
    try:
        impero_c = impero_utils.compile_gem(
            list(zip(variables, expressions)), ordering, remove_zeros=True)
    except impero_utils.NoopError:
        return 0
    return sum(
        numpy.prod([index.extent for index in impero_c.indices[temporary]],
                   dtype=int)
        for temporary in impero_c.temporaries)


def _factorise(
        pairs: tuple[tuple[Node, Node], ...],
        preserve_maps: bool) -> tuple[tuple[Node, MonomialSum], ...]:
    """Argument factorize and delta cancel one representation of the maps.

    Parameters
    ----------
    pairs
        Output variables and their integral expressions.
    preserve_maps
        Keep one-axis sums as finite element linear operands when true;
        expose their scalar polynomial structure when false.

    Returns
    -------
    tuple of tuple
        Output variables and their delta-cancelled monomial sums.

    """
    index_replacer = MemoizerArg(filtered_replace_indices)
    delta_inside = Memoizer(_delta_inside)
    narrow_variables = OrderedDict()
    delta_simplified = defaultdict(MonomialSum)

    groups = groupby(
        pairs, key=lambda pair: frozenset(pair[0].free_indices))
    for free_indices, pair_group in groups:
        variables, expressions = zip(*pair_group)
        argument_indices = set(free_indices)
        classifier = partial(
            classify, argument_indices, delta_inside=delta_inside)
        monomial_sums = collect_monomials(
            expressions, classifier,
            argument_indices if preserve_maps else ())
        for variable, monomial_sum in zip(variables, monomial_sums):
            for monomial in monomial_sum:
                var, indices, atomics, rest = delta_elimination(
                    variable, *monomial, index_replacer)
                narrow_variables.setdefault(var)
                delta_simplified[var].add(
                    indices, atomics, rest)

    return tuple((variable, delta_simplified[variable])
                 for variable in narrow_variables)


def _plans(
        assignments: tuple[tuple[Node, MonomialSum], ...],
        quadrature_indices: tuple) -> tuple[Plan, Plan]:
    """Place quadrature reductions, one plan per contraction strategy.

    Separate orderings minimize arithmetic; one shared ordering spans the
    assignments over a single loop nest, keeping the values they share
    narrow.  Both searches are exhaustive in the quadrature axes, of which
    a reference cell supplies one per direction.

    Parameters
    ----------
    assignments
        Output variables and their delta-cancelled monomial sums.
    quadrature_indices
        Every quadrature index of the integral, in source order.

    Returns
    -------
    separate
        Plan giving each assignment its cheapest ordering.
    shared
        Plan contracting every assignment in one common ordering.

    """
    contracted = []
    for _, monomial_sum in assignments:
        summed = set(chain.from_iterable(
            monomial.sum_indices for monomial in monomial_sum))
        contracted.append(
            tuple(index for index in quadrature_indices if index in summed))

    factorised = {
        (position, ordering): sum_factorise(variable, ordering, monomial_sum)
        for position, (variable, monomial_sum) in enumerate(assignments)
        for ordering in permutations(contracted[position])}

    def plan(orderings) -> Plan:
        return tuple(
            (variable, factorised[position, orderings[position]])
            for position, (variable, _) in enumerate(assignments))

    separate = plan([
        min(permutations(axes),
            key=lambda ordering, p=position: estimate_cost(
                (factorised[p, ordering],)))
        for position, axes in enumerate(contracted)])
    shared = min(
        (plan([tuple(index for index in ordering if index in axes)
               for axes in contracted])
         for ordering in permutations(quadrature_indices)),
        key=lambda candidate: estimate_cost(
            expression for _, expression in candidate))
    return separate, shared


def _select_plan(
        pairs: tuple[tuple[Node, Node], ...],
        quadrature_indices: tuple) -> Plan:
    """Minimize arithmetic among plans whose temporaries fit the budget.

    Preserving a linear map exposes tabulation reuse, expanding it exposes
    scalar factorization, and neither dominates.  Overflowing cache is a
    cliff rather than a gradient, so storage bounds the search instead of
    trading against arithmetic.  When no plan fits, take the narrowest.

    Parameters
    ----------
    pairs
        Output variables and their integral expressions.
    quadrature_indices
        Every quadrature index of the integral, in source order.

    Returns
    -------
    Plan
        Optimized output variables and GEM expressions.

    """
    candidates = list(dict.fromkeys(
        plan
        for preserve_maps in (False, True)
        for plan in _plans(_factorise(pairs, preserve_maps),
                           quadrature_indices)))
    if len(candidates) == 1:
        return candidates[0]

    storage = {plan: _declared_storage(plan, quadrature_indices)
               for plan in candidates}
    feasible = [plan for plan in candidates
                if storage[plan] <= storage_budget]
    if not feasible:
        return min(candidates, key=storage.get)
    return min(feasible,
               key=lambda plan: estimate_cost(
                   expression for _, expression in plan))


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

    return _select_plan(tuple(pairs), tuple(quadrature_indices))


finalise_options = dict(replace_delta=True, remove_componenttensors=False)


def classify(argument_indices, expression, delta_inside):
    """Classify one expression for multilinear factorization.

    Parameters
    ----------
    argument_indices : set of Index
        Free argument indices.
    expression : Node
        Expression to classify.
    delta_inside : callable
        Predicate detecting delta nodes.
    Returns
    -------
    str
        Refactorization label.
    """
    n = len(argument_indices.intersection(expression.free_indices))
    if n == 0:
        return OTHER
    elif n == 1:
        if isinstance(expression, Conditional):
            return ATOMIC
        if isinstance(expression, (Delta, Indexed)) \
                and not delta_inside(expression):
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

    assert set(var_indices) <= set(variable.free_indices)
    # A delta may replace a variable index by a contraction index.  That
    # index now describes a scatter in the assignment, not a sum.
    sum_indices = [i for i in sum_indices if i not in variable.free_indices]
    return variable, sum_indices, args, rest


def sum_factorise(variable, tail_ordering, monomial_sum):
    return optimise_monomial_sum(
        monomial_sum, variable.index_ordering(),
        tuple(tail_ordering))
