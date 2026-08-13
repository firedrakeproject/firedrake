from collections import OrderedDict, defaultdict, namedtuple
from collections.abc import Iterable
from functools import partial
from itertools import chain, zip_longest
import math

from gem.gem import Conditional, Delta, Index, Indexed, Sum, index_sum, one
from gem.node import Memoizer, MemoizerArg, traversal
from gem.optimise import filtered_replace_indices
from gem.optimise import delta_elimination as _delta_elimination
from gem.optimise import (
    estimate_cost, hoist_linear_index, replace_division,
    unroll_indexsum,
)
from gem.refactorise import (ATOMIC, COMPOUND, OTHER, MonomialSum,
                             collect_factorisation_plans)
from gem.unconcatenate import unconcatenate
from gem.coffee import optimise_monomial_sum
from gem.utils import groupby


Integral = namedtuple('Integral', ['expression',
                                   'quadrature_multiindex',
                                   'argument_indices'])


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


def _factorisation_candidates(
        expression, argument_indices,
        delta_inside) -> tuple[MonomialSum, ...]:
    """Build expanded and map-preserving factorisation plans.

    Parameters
    ----------
    expression : Node
        Multilinear integrand to factorize.
    argument_indices : set of Index
        Free argument indices.
    delta_inside : callable
        Memoized predicate detecting delta nodes.

    Returns
    -------
    tuple of MonomialSum
        Distinct contraction plans.
    """
    classifier = partial(
        classify, argument_indices, delta_inside=delta_inside)
    return collect_factorisation_plans(
        expression, classifier, argument_indices)


def _sum_factorisation_order(
        indices: Iterable[Index],
        monomial_sum: MonomialSum) -> tuple[Index, ...]:
    """Order quadrature contractions by their retained index support.

    A quadrature direction belongs earlier when its minimal dependency
    frontier retains fewer non-quadrature indices.  COFFEE can then isolate
    that contraction before introducing more strongly coupled factors.  The
    Ordinary tensor products have equal support sizes. They retain the
    extent-based stable ordering. Nested simplex factors naturally order
    themselves from least to most coupled.

    Parameters
    ----------
    indices : iterable of Index
        Quadrature indices to contract.
    monomial_sum : MonomialSum
        Factorized integrand.

    Returns
    -------
    tuple of Index
        Contraction indices from outermost to innermost stage.
    """
    indices = tuple(indices)
    contraction_indices = frozenset(indices)
    factors = tuple(
        factor
        for monomial in monomial_sum
        for factor in (*monomial.atomics, monomial.rest))
    nodes = tuple(traversal(factors))

    def support_size(index: Index) -> int:
        support = set()
        for node in nodes:
            if (index in node.free_indices
                    and not any(index in child.free_indices
                                for child in node.children)):
                support.update(
                    set(node.free_indices) - contraction_indices)
        return math.prod(argument.extent for argument in support)

    return tuple(sorted(
        indices, key=lambda index: (support_size(index), index.extent)))


def _optimise_candidate(
        variable, monomial_sum, quadrature_indices,
        index_replacer) -> tuple[tuple, ...]:
    """Apply delta elimination and contraction optimization to one plan.

    Parameters
    ----------
    variable : Node
        Assignment variable.
    monomial_sum : MonomialSum
        Candidate polynomial representation.
    quadrature_indices : tuple of Index
        Preferred quadrature contraction order.
    index_replacer : MemoizerArg
        Shared index-substitution mapper.

    Returns
    -------
    tuple
        Optimized assignment pairs for cost evaluation.
    """
    narrow_variables = OrderedDict()
    simplified = defaultdict(MonomialSum)
    for monomial in monomial_sum:
        var, indices, atomics, rest = delta_elimination(
            variable, *monomial, index_replacer)
        narrow_variables.setdefault(var)
        simplified[var].add(indices, atomics, rest)

    pairs = []
    for var in narrow_variables:
        candidate = simplified[var]
        contracted = set(chain.from_iterable(
            monomial.sum_indices for monomial in candidate))
        ordering = sorted(
            (index for index in quadrature_indices
             if index in contracted),
            key=lambda index: index.extent)
        pairs.append((
            var, sum_factorise(var, ordering, candidate)))
    return tuple(pairs)


def _candidate_score(pairs: tuple[tuple, ...]) -> tuple[int, ...]:
    """Estimate work and storage in a GEM contraction candidate.

    Parameters
    ----------
    pairs
        Assignment pairs to schedule.

    Returns
    -------
    tuple of int
        Operations, total contraction storage, largest contraction, and
        expression-node count. Lexicographic ordering prioritizes arithmetic
        work.
    """
    return estimate_cost(expression for _, expression in pairs)


def _select_factorisation_plan(
        variable, candidates, quadrature_indices,
        index_replacer) -> MonomialSum:
    """Choose the least-cost admitted factorisation plan.

    Parameters
    ----------
    variable : Node
        Assignment variable.
    candidates : tuple of MonomialSum
        Algebraically equivalent factorisation plans.
    quadrature_indices : tuple of Index
        Preferred quadrature contraction order.
    index_replacer : MemoizerArg
        Shared index-substitution mapper.

    Returns
    -------
    MonomialSum
        Selected factorisation plan.
    """
    return min(
        candidates,
        key=lambda candidate: _candidate_score(_optimise_candidate(
            variable, candidate, quadrature_indices, index_replacer)))


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
    quadrature_indices = tuple(quadrature_indices)
    # Group assignment pairs by argument indices
    for free_indices, pair_group in groupby(pairs, group_key):
        variables, expressions = zip(*pair_group)
        argument_indices = set(free_indices)
        for variable, expression in zip(variables, expressions):
            candidate_groups = _factorisation_candidates(
                expression, argument_indices, delta_inside)
            monomial_sum = _select_factorisation_plan(
                variable, candidate_groups, quadrature_indices,
                index_replacer)
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
        sum_indices = _sum_factorisation_order(
            sum_indices, monomial_sum)
        # Apply sum factorisation combined with COFFEE technology
        expression = sum_factorise(variable, sum_indices, monomial_sum)
        expression = hoist_linear_index(
            expression, variable.free_indices)
        yield (variable, expression)


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
