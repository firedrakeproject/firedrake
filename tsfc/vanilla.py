from __future__ import absolute_import, print_function, division
from six import iteritems

from gem import index_sum
from gem.optimise import unroll_indexsum


def integrate(expressions, quadrature_multiindex, parameters):
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        expressions = unroll_indexsum(expressions, predicate=predicate)
    return [index_sum(e, quadrature_multiindex) for e in expressions]


def aggregate(rep_dict):
    for variable, reps in iteritems(rep_dict):
        expressions = reps
        for expression in expressions:
            yield (variable, expression)
