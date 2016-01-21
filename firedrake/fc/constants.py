from __future__ import absolute_import

import numpy


NUMPY_TYPE = numpy.dtype("double")

PRECISION = numpy.finfo(NUMPY_TYPE).precision

SCALAR_TYPE = {numpy.dtype("double"): "double",
               numpy.dtype("float32"): "float"}[NUMPY_TYPE]


RESTRICTION_MAP = {"+": 0, "-": 1}
