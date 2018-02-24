import numpy

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

    "scalar_type": "double",
    # Precision of float printing (number of digits)
    "precision": numpy.finfo(numpy.dtype("double")).precision,

    "scalar_type": "double"
}


def default_parameters():
    return PARAMETERS.copy()

numpy_type_map = {"double": numpy.dtype("double"),
                  "float": numpy.dtype("float32"),
                  "double complex": numpy.dtype("complex128")}
