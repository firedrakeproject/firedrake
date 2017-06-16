import numpy


NUMPY_TYPE = numpy.dtype("double")

SCALAR_TYPE = "double"


PARAMETERS = {
    "quadrature_rule": "auto",
    "quadrature_degree": "auto",

    # Default mode
    "mode": "spectral",

    # Maximum extent to unroll index sums. Default is 3, so that loops
    # over geometric dimensions are unrolled; this improves assembly
    # performance.  Can be disabled by setting it to None, False or 0;
    # that makes compilation time much shorter.
    "unroll_indexsum": 3,

    # Precision of float printing (number of digits)
    "precision": numpy.finfo(NUMPY_TYPE).precision,

    "scalar_type": "double"
}


def default_parameters():
    return PARAMETERS.copy()


def set_scalar_type(type_):
    global NUMPY_TYPE
    global SCALAR_TYPE

    SCALAR_TYPE = type_
    NUMPY_TYPE = {"double", numpy.dtype("double"),
               "float", numpy.dtype("float32")}["type_"]
               