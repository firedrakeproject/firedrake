from __future__ import absolute_import, print_function, division
from six.moves import map, zip

from functools import partial
from itertools import chain

from gem import index_sum
from gem.optimise import delta_elimination as _delta_elimination
from gem.optimise import sum_factorise as _sum_factorise
from gem.optimise import unroll_indexsum
from gem.refactorise import ATOMIC, COMPOUND, OTHER, MonomialSum, collect_monomials


def delta_elimination(sum_indices, args, rest):
    factors = [rest] + list(args)
    sum_indices, factors = _delta_elimination(sum_indices, factors)
    rest = factors.pop(0)
    args = factors
    return sum_indices, args, rest


def sum_factorise(sum_indices, args, rest):
    sum_indices = list(sum_indices)
    sum_indices.reverse()
    factors = args + (rest,)
    return _sum_factorise(sum_indices, factors)


def Integrals(expressions, quadrature_multiindex, argument_multiindices, parameters):
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        expressions = unroll_indexsum(expressions, predicate=predicate)
    argument_indices = set(chain(*argument_multiindices))
    return [(argument_indices,
             index_sum(e, quadrature_multiindex))
            for e in expressions]


def flatten(var_reps):
    def classify(argument_indices, expression):
        n = len(argument_indices.intersection(expression.free_indices))
        if n == 0:
            return OTHER
        elif n == 1:
            return ATOMIC
        else:
            return COMPOUND

    for variable, reps in var_reps:
        argument_indicez, expressions = zip(*reps)
        argument_indices, = set(map(frozenset, argument_indicez))
        classifier = partial(classify, argument_indices)
        for monomial_sum in collect_monomials(expressions, classifier):
            delta_simplified = MonomialSum()
            for monomial in monomial_sum:
                delta_simplified.add(*delta_elimination(*monomial))

            for monomial in delta_simplified:
                yield (variable, sum_factorise(*monomial))


finalise_options = {}
