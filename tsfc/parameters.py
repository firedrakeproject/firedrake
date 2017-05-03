from __future__ import absolute_import, print_function, division

import numpy


NUMPY_TYPE = numpy.dtype("double")

SCALAR_TYPE = {numpy.dtype("double"): "double",
               numpy.dtype("float32"): "float"}[NUMPY_TYPE]


PARAMETERS = {
    "quadrature_rule": "auto",
    "quadrature_degree": "auto",

    # Default mode
    "mode": "coffee",

    # Maximum extent to unroll index sums. Default is 3, so that loops
    # over geometric dimensions are unrolled; this improves assembly
    # performance.  Can be disabled by setting it to None, False or 0;
    # that makes compilation time much shorter.
    "unroll_indexsum": 3,

    # Precision of float printing (number of digits)
    "precision": numpy.finfo(NUMPY_TYPE).precision,
}


def default_parameters():
    return PARAMETERS.copy()
