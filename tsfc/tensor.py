from collections import defaultdict
from functools import partial, reduce
from itertools import count

import numpy

import gem
from gem.optimise import remove_componenttensors, unroll_indexsum
from gem.refactorise import ATOMIC, COMPOUND, OTHER, collect_monomials
from gem.unconcatenate import flatten as concatenate


def einsum(factors, sum_indices):
    """Evaluates a tensor product at compile time.

    :arg factors: iterable of indexed GEM literals
    :arg sum_indices: indices to sum over
    :returns: a single indexed GEM literal
    """
    # Maps the problem onto numpy.einsum
    index2letter = defaultdict(partial(lambda c: chr(ord('i') + next(c)), count()))
    operands = []
    subscript_parts = []
    for factor in factors:
        literal, = factor.children
        selectors = []
        letters = []
        for index in factor.multiindex:
            if isinstance(index, int):
                selectors.append(index)
            else:
                selectors.append(slice(None))
                letters.append(index2letter[index])
        operands.append(literal.array.__getitem__(tuple(selectors)))
        subscript_parts.append(''.join(letters))

    result_pairs = sorted((letter, index)
                          for index, letter in index2letter.items()
                          if index not in sum_indices)

    subscripts = ','.join(subscript_parts) + '->' + ''.join(l for l, i in result_pairs)
    tensor = numpy.einsum(subscripts, *operands)
    return gem.Indexed(gem.Literal(tensor), tuple(i for l, i in result_pairs))


def Integrals(expressions, quadrature_multiindex, argument_multiindices, parameters):
    # Concatenate
    expressions = concatenate(expressions)

    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        expressions = unroll_indexsum(expressions, predicate=predicate)

    # Refactorise
    def classify(quadrature_indices, expression):
        if not quadrature_indices.intersection(expression.free_indices):
            return OTHER
        elif isinstance(expression, gem.Indexed) and isinstance(expression.children[0], gem.Literal):
            return ATOMIC
        else:
            return COMPOUND
    classifier = partial(classify, set(quadrature_multiindex))

    result = []
    for expr, monomial_sum in zip(expressions, collect_monomials(expressions, classifier)):
        # Select quadrature indices that are present
        quadrature_indices = set(index for index in quadrature_multiindex
                                 if index in expr.free_indices)

        products = []
        for sum_indices, factors, rest in monomial_sum:
            # Collapse quadrature literals for each monomial
            if factors or quadrature_indices:
                replacement = einsum(remove_componenttensors(factors), quadrature_indices)
            else:
                replacement = gem.Literal(1)
            # Rebuild expression
            products.append(gem.IndexSum(gem.Product(replacement, rest), sum_indices))
        result.append(reduce(gem.Sum, products, gem.Zero()))
    return result


def flatten(var_reps, index_cache):
    for variable, reps in var_reps:
        expressions = reps
        for expression in expressions:
            yield (variable, expression)


finalise_options = {}
