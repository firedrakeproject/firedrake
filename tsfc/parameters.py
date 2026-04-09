import numpy
from loopy.target.c import CWithGNULibcTarget

try:
    from petsc4py.PETSc import ScalarType as _PETScScalarType
except ImportError:
    _PETScScalarType = numpy.float64


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

    # Scalar type numpy dtype — derived from PETSc compile-time precision
    "scalar_type": numpy.dtype(_PETScScalarType),

    # C type string matching scalar_type
    "scalar_type_c": {numpy.dtype(numpy.float32): "float",
                      numpy.dtype(numpy.float64): "double",
                      numpy.dtype(numpy.complex128): "double complex",
                      numpy.dtype(numpy.complex64): "float complex",
                      }.get(numpy.dtype(_PETScScalarType), "double"),

    # Whether to wrap the generated kernels in a PETSc event
    "add_petsc_events": False,
}


target = CWithGNULibcTarget()


def default_parameters():
    return PARAMETERS.copy()


def is_complex(scalar_type):
    """Decides complex mode based on scalar type."""
    return scalar_type and (isinstance(scalar_type, numpy.dtype) and scalar_type.kind == 'c') \
        or (isinstance(scalar_type, str) and "complex" in scalar_type)
