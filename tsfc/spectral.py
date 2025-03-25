from collections import OrderedDict, defaultdict, namedtuple
from functools import partial, reduce
from itertools import chain, zip_longest

from gem.gem import Delta, Indexed, Sum, index_sum, one
from gem.node import Memoizer
from gem.optimise import delta_elimination as _delta_elimination
from gem.optimise import remove_componenttensors, replace_division, unroll_indexsum
from gem.refactorise import ATOMIC, COMPOUND, OTHER, MonomialSum, collect_monomials
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
    argument_indices = tuple(chain(*argument_multiindices))
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
        pairs.append((variable, reduce(Sum, expressions)))

        # Collect quadrature_indices
        for r in reps:
            quadrature_indices.update(zip_longest(r.quadrature_multiindex, ()))

    # Split Concatenate nodes
    pairs = unconcatenate(pairs, cache=index_cache)

    def group_key(pair):
        variable, expression = pair
        return frozenset(variable.free_indices)

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
                var, s, a, r = delta_elimination(variable, *monomial)
                narrow_variables.setdefault(var)
                delta_simplified[var].add(s, a, r)

    # Final factorisation
    for variable in narrow_variables:
        monomial_sum = delta_simplified[variable]
        # Collect sum indices applicable to the current MonomialSum
        sum_indices = set().union(*[m.sum_indices for m in monomial_sum])
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


finalise_options = dict(replace_delta=False)


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


def delta_elimination(variable, sum_indices, args, rest):
    """IndexSum-Delta cancellation for monomials."""
    factors = list(args) + [variable, rest]  # construct factors

    def prune(factors):
        # Skip last factor (``rest``, see above) which can be
        # arbitrarily complicated, so its pruning may be expensive,
        # and its early pruning brings no advantages.
        result = remove_componenttensors(factors[:-1])
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
