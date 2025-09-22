from functools import reduce

from gem import index_sum, Sum
from gem.optimise import unroll_indexsum
from gem.unconcatenate import unconcatenate


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


def flatten(var_reps, index_cache):
    """Flatten mode-specific intermediate representation to a series of
    assignments.

    :arg var_reps: series of (return variable, [integral representation]) pairs
    :arg index_cache: cache :py:class:`dict` for :py:func:`unconcatenate`

    :returns: series of (return variable, GEM expression root) pairs
    """
    return unconcatenate([(variable, reduce(Sum, reps))
                          for variable, reps in var_reps],
                         cache=index_cache)


finalise_options = dict(remove_componenttensors=False)
"""To avoid duplicate work, these options that are safe to pass to
:py:func:`gem.impero_utils.preprocess_gem`."""
