from __future__ import absolute_import, print_function, division

import numpy
import itertools
from functools import partial
from six import iteritems, iterkeys
from six.moves import filter
from collections import defaultdict
from gem.optimise import (replace_division, make_sum, make_product,
                          unroll_indexsum, replace_delta, remove_componenttensors)
from gem.refactorise import Monomial, ATOMIC, COMPOUND, OTHER, collect_monomials
from gem.node import traversal
from gem.gem import Indexed, IndexSum, Failure, one, index_sum
from gem.utils import groupby


import tsfc.vanilla as vanilla

flatten = vanilla.flatten

finalise_options = dict(replace_delta=False, remove_componenttensors=False)


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
    # Choose GEM expression as the integral representation
    expressions = [index_sum(e, quadrature_multiindex) for e in expressions]
    expressions = replace_delta(expressions)
    expressions = remove_componenttensors(expressions)
    expressions = replace_division(expressions)
    argument_indices = tuple(itertools.chain(*argument_multiindices))
    return optimise_expressions(expressions, argument_indices)


def optimise_expressions(expressions, argument_indices):
    """Perform loop optimisations on GEM DAGs

    :arg expressions: list of GEM DAGs
    :arg argument_indices: tuple of argument indices

    :returns: list of optimised GEM DAGs
    """
    # Skip optimisation for if Failure node is present
    for n in traversal(expressions):
        if isinstance(n, Failure):
            return expressions

    def classify(argument_indices, expression):
        n = len(argument_indices.intersection(expression.free_indices))
        if n == 0:
            return OTHER
        elif n == 1:
            if isinstance(expression, Indexed):
                return ATOMIC
            else:
                return COMPOUND
        else:
            return COMPOUND

    # Apply argument factorisation unconditionally
    classifier = partial(classify, set(argument_indices))
    monomial_sums = collect_monomials(expressions, classifier)
    return [optimise_monomial_sum(ms, argument_indices) for ms in monomial_sums]


def index_extent(factor, argument_indices):
    """Compute the product of the extents of argument indices of a GEM expression

    :arg factor: GEM expression
    :arg argument_indices: set of argument indices

    :returns: product of extents of argument indices
    """
    return numpy.product([i.extent for i in set(factor.free_indices).intersection(argument_indices)])


def monomial_sum_to_expression(monomial_sum):
    """Convert a monomial sum to a GEM expression.

    :arg monomial_sum: an iterable of :class:`Monomial`s

    :returns: GEM expression
    """
    indexsums = []  # The result is summation of indexsums
    # Group monomials according to their sum indices
    groups = groupby(monomial_sum, key=lambda m: frozenset(m.sum_indices))
    # Create IndexSum's from each monomial group
    for _, monomials in groups:
        sum_indices = monomials[0].sum_indices
        products = [make_product(monomial.atomics + (monomial.rest,)) for monomial in monomials]
        indexsums.append(IndexSum(make_sum(products), sum_indices))
    return make_sum(indexsums)


def find_optimal_atomics(monomials, argument_indices):
    """Find optimal atomic common subexpressions, which produce least number of
    terms in the resultant IndexSum when factorised.

    :arg monomials: An iterable of :class:`Monomial`s, all of which should have
                    the same sum indices
    :arg argument_indices: tuple of argument indices

    :returns: list of atomic GEM expressions
    """
    atomic_index = defaultdict(partial(next, itertools.count()))  # atomic -> int
    connections = []
    # add connections (list of sets, items in each tuple form a product)
    for monomial in monomials:
        connections.append(set(map(lambda a: atomic_index[a], monomial.atomics)))

    num_atomics = len(atomic_index)
    if num_atomics <= 1:
        return tuple(iterkeys(atomic_index))

    # 0-1 integer programming
    def calculate_extent(solution):
        extent = 0
        for atomic, index in iteritems(atomic_index):
            if index in solution:
                extent += index_extent(atomic, argument_indices)
        return extent

    def solve_ip(idx, solution, optimal_solution):
        if idx >= num_atomics:
            return
        if len(solution) >= len(optimal_solution):
            return
        solution.add(idx)
        feasible = True
        for conn in connections:
            if not solution.intersection(conn):
                feasible = False
                break
        if feasible:
            if len(solution) < len(optimal_solution) or \
                    (len(solution) == len(optimal_solution) and calculate_extent(solution) > calculate_extent(optimal_solution)):
                optimal_solution.clear()
                optimal_solution.update(solution)
            # No need to search further
        else:
            solve_ip(idx + 1, solution, optimal_solution)
        solution.remove(idx)
        solve_ip(idx + 1, solution, optimal_solution)

    optimal_solution = set(range(num_atomics))  # start by choosing all atomics
    solution = set()
    solve_ip(0, solution, optimal_solution)

    def optimal(atomic):
        return atomic_index[atomic] in optimal_solution

    return tuple(sorted(filter(optimal, atomic_index), key=atomic_index.get))


def factorise_atomics(monomials, optimal_atomics, argument_indices):
    """Group and factorise monomials using a list of atomics as common
    subexpressions. Create new monomials for each group and optimise them recursively.

    :arg monomials: an iterable of :class:`Monomial`s, all of which should have
                    the same sum indices
    :arg optimal_atomics: list of tuples of atomics to be used as common subexpression
    :arg argument_indices: tuple of argument indices

    :returns: an iterable of :class:`Monomials`s after factorisation
    """
    if not optimal_atomics or len(monomials) <= 1:
        return monomials

    # Group monomials with respect to each optimal atomic
    def group_key(monomial):
        for oa in optimal_atomics:
            if oa in monomial.atomics:
                return oa
        assert False, "Expect at least one optimal atomic per monomial."
    factor_group = groupby(monomials, key=group_key)

    # We should not drop monomials
    assert sum(len(ms) for _, ms in factor_group) == len(monomials)

    sum_indices = next(iter(monomials)).sum_indices
    new_monomials = []
    for oa, monomials in factor_group:
        # Create new MonomialSum for the factorised out terms
        sub_monomials = []
        for monomial in monomials:
            atomics = list(monomial.atomics)
            atomics.remove(oa)  # remove common factor
            sub_monomials.append(Monomial((), tuple(atomics), monomial.rest))
        # Continue to factorise the remaining expression
        sub_monomials = optimise_monomials(sub_monomials, argument_indices)
        if len(sub_monomials) == 1:
            # Factorised part is a product, we add back the common atomics then
            # add to new MonomialSum directly rather than forming a product node
            # Retaining the monomial structure enables applying associativity
            # when forming GEM nodes later.
            sub_monomial, = sub_monomials
            new_monomials.append(
                Monomial(sum_indices, (oa,) + sub_monomial.atomics, sub_monomial.rest))
        else:
            # Factorised part is a summation, we need to create a new GEM node
            # and multiply with the common factor
            node = monomial_sum_to_expression(sub_monomials)
            # If the free indices of the new node intersect with argument indices,
            # add to the new monomial as `atomic`, otherwise add as `rest`.
            # Note: we might want to continue to factorise with the new atomics
            # by running optimise_monoials twice.
            if set(argument_indices) & set(node.free_indices):
                new_monomials.append(Monomial(sum_indices, (oa, node), one))
            else:
                new_monomials.append(Monomial(sum_indices, (oa, ), node))
    return new_monomials


def optimise_monomial_sum(monomial_sum, argument_indices):
    """Choose optimal common atomic subexpressions and factorise a
    :class:`MonomialSum` object to create a GEM expression.

    :arg monomial_sum: a :class:`MonomialSum` object
    :arg argument_indices: tuple of argument indices

    :returns: factorised GEM expression
    """
    groups = groupby(monomial_sum, key=lambda m: frozenset(m.sum_indices))
    new_monomials = []
    for _, monomials in groups:
        new_monomials.extend(optimise_monomials(monomials, argument_indices))
    return monomial_sum_to_expression(new_monomials)


def optimise_monomials(monomials, argument_indices):
    """Choose optimal common atomic subexpressions and factorise an iterable
    of monomials.

    :arg monomials: an iterable of :class:`Monomial`s, all of which should have
                    the same sum indices
    :arg argument_indices: tuple of argument indices

    :returns: an iterable of factorised :class:`Monomials`s
    """
    assert len(set(frozenset(m.sum_indices) for m in monomials)) <= 1,\
        "All monomials required to have same sum indices for factorisation"

    optimal_atomics = find_optimal_atomics(monomials, argument_indices)
    return factorise_atomics(monomials, optimal_atomics, argument_indices)
