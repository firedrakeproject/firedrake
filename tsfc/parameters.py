import warnings
import numpy
from loopy.target.c import CWithGNULibcTarget

try:
    from petsc4py.PETSc import ScalarType as _PETScScalarType
except ImportError:
    warnings.warn(
        "petsc4py not found; defaulting TSFC scalar_type to float64. "
        "This will produce incorrect kernels for non-double PETSc builds.",
        RuntimeWarning,
        stacklevel=1,
    )
    _PETScScalarType = numpy.float64

_dtype_to_c = {
    numpy.dtype(numpy.float32): "float",
    numpy.dtype(numpy.float64): "double",
    numpy.dtype(numpy.complex128): "double complex",
    numpy.dtype(numpy.complex64): "float complex",
}
_scalar_dtype = numpy.dtype(_PETScScalarType)
if _scalar_dtype not in _dtype_to_c:
    raise ValueError(
        f"Unsupported PETSc scalar type {_PETScScalarType!r}; "
        f"expected one of {list(_dtype_to_c)}"
    )

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
    "scalar_type": _scalar_dtype,

    # C type string matching scalar_type
    "scalar_type_c": _dtype_to_c[_scalar_dtype],

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
