from __future__ import absolute_import

import numpy


NUMPY_TYPE = numpy.dtype("double")

PRECISION = numpy.finfo(NUMPY_TYPE).precision

SCALAR_TYPE = {numpy.dtype("double"): "double",
               numpy.dtype("float32"): "float"}[NUMPY_TYPE]


PARAMETERS = {
    "quadrature_rule": "auto",
    "quadrature_degree": "auto",

    # Trust COFFEE to do loop-invariant code motion. Disabled by
    # default as COFFEE does not work on TSFC-generated code yet.
    # When enabled, it allows the inlining of expressions even if that
    # pulls calculations into inner loops.
    "coffee_licm": False,

    # Maximum extent to unroll index sums. Default is 3, so that loops
    # over geometric dimensions are unrolled; this improves assembly
    # performance.  Can be disabled by setting it to None, False or 0;
    # that makes compilation time much shorter.
    "unroll_indexsum": 3,
}


def default_parameters():
    return PARAMETERS.copy()
