from __future__ import absolute_import, print_function, division

from gem import index_sum
from gem.optimise import unroll_indexsum


def Integrals(expressions, quadrature_multiindex, argument_multiindices, parameters):
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        expressions = unroll_indexsum(expressions, predicate=predicate)
    return [index_sum(e, quadrature_multiindex) for e in expressions]


def flatten(var_reps):
    for variable, reps in var_reps:
        expressions = reps
        for expression in expressions:
            yield (variable, expression)
