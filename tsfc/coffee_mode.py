from functools import partial, reduce

from gem.node import traversal, Memoizer
from gem.gem import Failure, Sum, index_sum
from gem.optimise import replace_division, unroll_indexsum
from gem.refactorise import collect_monomials
from gem.unconcatenate import unconcatenate
from gem.coffee import optimise_monomial_sum
from gem.utils import groupby

import tsfc.spectral as spectral


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


def flatten(var_reps, index_cache):
    """Flatten mode-specific intermediate representation to a series of
    assignments.

    :arg var_reps: series of (return variable, [integral representation]) pairs
    :arg index_cache: cache :py:class:`dict` for :py:func:`unconcatenate`

    :returns: series of (return variable, GEM expression root) pairs
    """
    assignments = unconcatenate([(variable, reduce(Sum, reps))
                                 for variable, reps in var_reps],
                                cache=index_cache)

    def group_key(assignment):
        variable, expression = assignment
        return variable.free_indices

    for argument_indices, assignment_group in groupby(assignments, group_key):
        variables, expressions = zip(*assignment_group)
        expressions = optimise_expressions(expressions, argument_indices)
        for var, expr in zip(variables, expressions):
            yield (var, expr)


finalise_options = dict(remove_componenttensors=False)


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

    # Apply argument factorisation unconditionally
    classifier = partial(spectral.classify, set(argument_indices),
                         delta_inside=Memoizer(spectral._delta_inside))
    monomial_sums = collect_monomials(expressions, classifier)
    return [optimise_monomial_sum(ms, argument_indices) for ms in monomial_sums]
