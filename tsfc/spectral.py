from __future__ import absolute_import, print_function, division
from six.moves import zip

from functools import partial, reduce

from gem import Delta, Indexed, Sum, index_sum
from gem.optimise import delta_elimination as _delta_elimination
from gem.optimise import sum_factorise as _sum_factorise
from gem.optimise import replace_division, unroll_indexsum
from gem.refactorise import ATOMIC, COMPOUND, OTHER, MonomialSum, collect_monomials
from gem.unconcatenate import unconcatenate
from gem.utils import groupby


def delta_elimination(sum_indices, args, rest):
    """IndexSum-Delta cancellation for monomials."""
    factors = [rest] + list(args)  # construct factors
    sum_indices, factors = _delta_elimination(sum_indices, factors)
    # Destructure factors after cancellation
    rest = factors.pop(0)
    args = factors
    return sum_indices, args, rest


def sum_factorise(sum_indices, args, rest):
    """Optimised monomial product construction through sum factorisation
    with reversed sum indices."""
    sum_indices = list(sum_indices)
    sum_indices.reverse()
    factors = args + (rest,)
    return _sum_factorise(sum_indices, factors)


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
    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        expressions = unroll_indexsum(expressions, predicate=predicate)
    # Integral representation: just a GEM expression
    return replace_division([index_sum(e, quadrature_multiindex) for e in expressions])


def classify(argument_indices, expression):
    """Classifier for argument factorisation"""
    n = len(argument_indices.intersection(expression.free_indices))
    if n == 0:
        return OTHER
    elif n == 1:
        if isinstance(expression, (Delta, Indexed)):
            return ATOMIC
        else:
            return COMPOUND
    else:
        return COMPOUND


def flatten(var_reps, index_cache):
    assignments = unconcatenate([(variable, reduce(Sum, reps))
                                 for variable, reps in var_reps],
                                cache=index_cache)

    def group_key(assignment):
        variable, expression = assignment
        return variable.free_indices

    for free_indices, assignment_group in groupby(assignments, group_key):
        variables, expressions = zip(*assignment_group)
        classifier = partial(classify, set(free_indices))
        monomial_sums = collect_monomials(expressions, classifier)
        for variable, monomial_sum in zip(variables, monomial_sums):
            # Compact MonomialSum after IndexSum-Delta cancellation
            delta_simplified = MonomialSum()
            for monomial in monomial_sum:
                delta_simplified.add(*delta_elimination(*monomial))

            # Yield assignments
            for monomial in delta_simplified:
                yield (variable, sum_factorise(*monomial))


finalise_options = dict(remove_componenttensors=False)
