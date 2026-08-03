from collections import OrderedDict, defaultdict, namedtuple
from functools import partial
import itertools
from itertools import chain, zip_longest

from gem.gem import Conditional, Delta, Indexed, IndexSum, Sum, index_sum, one
from gem.node import Memoizer, MemoizerArg
from gem.optimise import filtered_replace_indices
from gem.optimise import delta_elimination as _delta_elimination
from gem.optimise import (
    estimate_cost, factorisation_group_options, hoist_linear_index,
    replace_division,
    unroll_indexsum,
)
from gem.refactorise import ATOMIC, COMPOUND, OTHER, MonomialSum, collect_monomials
from gem.unconcatenate import unconcatenate
from gem.coffee import sum_factorise_monomial_sum
from gem.utils import groupby


Integral = namedtuple('Integral', ['expression',
                                   'quadrature_multiindex',
                                   'argument_indices'])
FactorisationCandidates = namedtuple(
    'FactorisationCandidates', ['alternatives', 'baseline'])


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
        expression, argument_indices, quadrature_indices,
        delta_inside) -> tuple[FactorisationCandidates, ...]:
    """Build alternative pre-expansion grouping plans.

    Parameters
    ----------
    expression : Node
        Multilinear integrand to factorize.
    argument_indices : set of Index
        Free argument indices.
    quadrature_indices : tuple of Index
        Indices contracted by quadrature.
    delta_inside : callable
        Memoized predicate detecting delta nodes.

    Returns
    -------
    tuple of FactorisationCandidates
        Distinct monomial representations for each independent group of
        choices, together with the all-or-nothing grouping plans.
    """
    terms = factorisation_group_options(
        expression, argument_indices)
    layouts = OrderedDict()
    for term, layout, options in terms:
        option_sums = layouts.setdefault(
            layout, [MonomialSum() for _ in options])
        assert len(option_sums) == len(options)
        for position, groups in enumerate(options):
            classifier = partial(
                classify, argument_indices, quadrature_indices,
                delta_inside=delta_inside, groups=groups)
            monomial_sum, = collect_monomials([term], classifier)
            option_sums[position] = MonomialSum.sum(
                option_sums[position], monomial_sum)

    options = tuple(tuple(option_sums) for option_sums in layouts.values())
    supports = tuple(
        frozenset(
            atomic
            for option in option_sums
            for monomial in option
            for atomic in monomial.atomics)
        for option_sums in options)

    remaining = set(range(len(options)))
    components = []
    while remaining:
        component = {remaining.pop()}
        support = set().union(*(supports[i] for i in component))
        while True:
            neighbours = {
                i for i in remaining if supports[i].intersection(support)}
            if not neighbours:
                break
            component.update(neighbours)
            remaining.difference_update(neighbours)
            support.update(*(supports[i] for i in neighbours))
        components.append(tuple(sorted(component)))

    result = []
    for component in components:
        candidates = OrderedDict()
        baseline = OrderedDict()
        choices = (
            tuple(enumerate(options[i])) for i in component)
        for selected in itertools.product(*choices):
            positions, monomial_sums = zip(*selected)
            candidate = MonomialSum.sum(*monomial_sums)
            key = tuple(candidate)
            candidates.setdefault(key, candidate)
            if all(position in {0, len(options[i]) - 1}
                   for i, position in zip(component, positions)):
                baseline.setdefault(key, candidate)
        result.append(FactorisationCandidates(
            tuple(candidates.values()), tuple(baseline.values())))
    return tuple(result)


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
        variable, candidate_groups, quadrature_indices,
        index_replacer) -> MonomialSum:
    """Choose a minimum-work plan under the prior DAG-size budget.

    The all-or-nothing grouping choices define the complexity budget of the
    previous factorisation algorithm. Independent groups are combined with a
    dynamic program, retaining the least expensive plan for each total DAG
    size. This exposes partial groupings without allowing expression growth.

    Parameters
    ----------
    variable : Node
        Assignment variable.
    candidate_groups : tuple of FactorisationCandidates
        Independent factorisation choices.
    quadrature_indices : tuple of Index
        Preferred quadrature contraction order.
    index_replacer : MemoizerArg
        Shared index-substitution mapper.

    Returns
    -------
    MonomialSum
        Selected factorisation plan.
    """
    plans = []
    node_budget = 0
    for group in candidate_groups:
        alternatives = []
        scores = {}
        for candidate in group.alternatives:
            score = _candidate_score(_optimise_candidate(
                variable, candidate, quadrature_indices, index_replacer))
            alternatives.append((score, candidate))
            scores[tuple(candidate)] = score
        plans.append(alternatives)
        node_budget += min(
            (scores[tuple(candidate)] for candidate in group.baseline),
            key=lambda score: score)[3]

    # score -> operations, total storage, largest intermediate, DAG nodes
    states = {(0, 0, 0, 0): ()}
    for alternatives in plans:
        updated = {}
        for left, selected in states.items():
            for right, candidate in alternatives:
                score = (
                    left[0] + right[0],
                    left[1] + right[1],
                    max(left[2], right[2]),
                    left[3] + right[3],
                )
                if score[3] <= node_budget:
                    previous = updated.get(score[3])
                    if previous is None or score < previous[0]:
                        updated[score[3]] = (score, selected + (candidate,))

        # A state with both a larger DAG and a worse lexicographic cost can
        # never become optimal as subsequent component costs are additive.
        states = {}
        best = None
        for size in sorted(updated):
            score, selected = updated[size]
            if best is None or score[:3] < best:
                states[score] = selected
                best = score[:3]

    score = min(states)
    return MonomialSum.sum(*states[score])


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
                expression, argument_indices, quadrature_indices,
                delta_inside)
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
        # Sort for increasing index extent, this obtains the good
        # factorisation for triangle x interval cells.  Python sort is
        # stable, so in the common case when index extents are equal,
        # the previous deterministic ordering applies which is good
        # for getting smaller temporaries.
        sum_indices = sorted(sum_indices, key=lambda index: index.extent)
        # Apply sum factorisation combined with COFFEE technology
        expression = sum_factorise(variable, sum_indices, monomial_sum)
        expression = hoist_linear_index(
            expression, variable.free_indices)
        yield (variable, expression)


finalise_options = dict(replace_delta=True, remove_componenttensors=False)


def classify(argument_indices, quadrature_indices, expression, delta_inside,
             groups=frozenset()):
    """Classify one expression for multilinear factorization.

    Parameters
    ----------
    argument_indices : set of Index
        Free argument indices.
    quadrature_indices : tuple of Index
        Indices contracted by quadrature.
    expression : Node
        Expression to classify.
    delta_inside : callable
        Predicate detecting delta nodes.
    groups : frozenset of Node
        Algebraic groups selected by contraction-plan optimization.

    Returns
    -------
    str
        Refactorization label.
    """
    if expression in groups:
        return ATOMIC
    n = len(argument_indices.intersection(expression.free_indices))
    if n == 0:
        return OTHER
    elif n == 1:
        if isinstance(expression, IndexSum) and set(
                expression.multiindex).isdisjoint(quadrature_indices):
            return ATOMIC
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
    return sum_factorise_monomial_sum(
        monomial_sum, tuple(tail_ordering), variable.index_ordering())
