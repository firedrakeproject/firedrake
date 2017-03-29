from __future__ import absolute_import, print_function, division

from gem import index_sum
from gem.optimise import unroll_indexsum


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
    return [index_sum(e, quadrature_multiindex) for e in expressions]


def flatten(var_reps):
    """Flatten mode-specific intermediate representation to a series of
    assignments.

    :arg var_reps: series of (return variable, [integral representation]) pairs

    :returns: series of (return variable, GEM expression root) pairs
    """
    for variable, reps in var_reps:
        expressions = reps  # representations are expressions
        for expression in expressions:
            yield (variable, expression)


finalise_options = {}
"""To avoid duplicate work, these options that are safe to pass to
:py:func:`gem.impero_utils.preprocess_gem`."""
