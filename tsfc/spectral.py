from __future__ import absolute_import, print_function, division
from six.moves import map, zip

from functools import partial
from itertools import chain

from gem import Delta, Indexed, index_sum
from gem.optimise import delta_elimination as _delta_elimination
from gem.optimise import sum_factorise as _sum_factorise
from gem.optimise import unroll_indexsum
from gem.refactorise import ATOMIC, COMPOUND, OTHER, MonomialSum, collect_monomials


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
    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        expressions = unroll_indexsum(expressions, predicate=predicate)
    # Integral representation: pair with the set of argument indices
    # and a GEM expression
    argument_indices = set(chain(*argument_multiindices))
    return [(argument_indices,
             index_sum(e, quadrature_multiindex))
            for e in expressions]


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


def flatten(var_reps):
    for variable, reps in var_reps:
        # Destructure representation
        argument_indicez, expressions = zip(*reps)
        # Assert identical argument indices for all integrals
        argument_indices, = set(map(frozenset, argument_indicez))
        # Argument factorise
        classifier = partial(classify, argument_indices)
        for monomial_sum in collect_monomials(expressions, classifier):
            # Compact MonomialSum after IndexSum-Delta cancellation
            delta_simplified = MonomialSum()
            for monomial in monomial_sum:
                delta_simplified.add(*delta_elimination(*monomial))

            # Yield assignments
            for monomial in delta_simplified:
                yield (variable, sum_factorise(*monomial))


finalise_options = dict(remove_componenttensors=False)
