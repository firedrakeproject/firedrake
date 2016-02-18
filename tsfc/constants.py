from __future__ import absolute_import

import numpy


NUMPY_TYPE = numpy.dtype("double")

PRECISION = numpy.finfo(NUMPY_TYPE).precision

SCALAR_TYPE = {numpy.dtype("double"): "double",
               numpy.dtype("float32"): "float"}[NUMPY_TYPE]


PARAMETERS = {
    "quadrature_rule": "auto",
    "quadrature_degree": "auto",
    "coffee_licm": False,
    "unroll_indexsum": 3,
}


def default_parameters():
    return PARAMETERS.copy()
