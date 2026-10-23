import collections
import ctypes
import functools

import loopy as lp
import numpy
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

IntType = numpy.dtype(PETSc.IntType)
RealType = numpy.dtype(PETSc.RealType)
ScalarType = numpy.dtype(PETSc.ScalarType)


DTypeLimit = collections.namedtuple("DTypeLimit", ["min", "max"])


_MPI_types = {}


def get_mpi_dtype(numpy_dtype, cdim=1):
    """Get an MPI datatype corresponding to a Dat.

    This builds (if necessary a contiguous derived datatype of the
    correct size).

    Also returns if it is a builtin type.
    """
    key = (numpy_dtype, cdim)
    try:
        return _MPI_types[key]
    except KeyError:
        tdict = MPI._typedict
        try:
            btype = tdict[numpy_dtype.char]
        except KeyError:
            raise RuntimeError("Unknown base type %r", numpy_dtype)
        if cdim == 1:
            typ = btype
            builtin = True
        else:
            typ = btype.Create_contiguous(cdim)
            typ.Commit()
            builtin = False
        return _MPI_types.setdefault(key, (typ, builtin))


# This is some magic lifted from PyOP2. It lets us do reductions using complex dtypes.
_numpy_types = {}


def _get_dtype(datatype):
    """Get a numpy datatype corresponding to an MPI datatype.

    Only works for contiguous datatypes."""
    try:
        # possibly unsafe if handles are recycled, but OK, because we
        # hold on to the contig types
        return _numpy_types[datatype.py2f()]
    except KeyError:
        base, combiner, _ = datatype.decode()
        while combiner == "DUP":
            base, combiner, _ = base.decode()
        # Allow for "NAMED", too, for complex scalar {MAX, MIN}.
        # if not (combiner == "CONTIGUOUS" or (complex_mode and combiner == "NAMED")):
        #     raise RuntimeError(
        #         f"Can only handle contiguous types or named types for complex scalar: "
        #         f"found combiner={combiner}"
        #     )
        try:
            tdict = MPI.__TypeDict__
        except AttributeError:
            tdict = MPI._typedict
        tdict = dict((v.py2f(), k) for k, v in tdict.items())
        try:
            base = tdict[base.py2f()]
        except KeyError:
            raise RuntimeError("Unhandled base datatype %r", base)
        return _numpy_types.setdefault(datatype.py2f(), base)


def reduction_op(op, invec, inoutvec, datatype):
    dtype = _get_dtype(datatype)
    invec = numpy.frombuffer(invec, dtype=dtype)
    inoutvec = numpy.frombuffer(inoutvec, dtype=dtype)
    inoutvec[:] = op(invec, inoutvec)


_contig_min_op = MPI.Op.Create(functools.partial(reduction_op, numpy.minimum), commute=True)
_contig_max_op = MPI.Op.Create(functools.partial(reduction_op, numpy.maximum), commute=True)


def as_cstr(dtype):
    """Convert a numpy dtype like object to a C type as a string."""
    return {"bool": "unsigned char",
            "int": "int",
            "int8": "int8_t",
            "int16": "int16_t",
            "int32": "int32_t",
            "int64": "int64_t",
            "uint8": "uint8_t",
            "uint16": "uint16_t",
            "uint32": "uint32_t",
            "uint64": "uint64_t",
            "float32": "float",
            "float64": "double",
            "complex128": "double complex"}[numpy.dtype(dtype).name]


def as_ctypes(dtype):
    """Convert a numpy dtype like object to a ctypes type."""
    return {"bool": ctypes.c_bool,
            "int": ctypes.c_int,
            "int8": ctypes.c_char,
            "int16": ctypes.c_int16,
            "int32": ctypes.c_int32,
            "int64": ctypes.c_int64,
            "uint8": ctypes.c_ubyte,
            "uint16": ctypes.c_uint16,
            "uint32": ctypes.c_uint32,
            "uint64": ctypes.c_uint64,
            "float32": ctypes.c_float,
            "float64": ctypes.c_double}[numpy.dtype(dtype).name]


def as_numpy_dtype(dtype):
    """Convert a dtype-like object into a numpy dtype."""
    if isinstance(dtype, numpy.dtype):
        return dtype
    elif isinstance(dtype, lp.types.NumpyType):
        return dtype.numpy_dtype
    else:
        raise ValueError


def dtype_limits(dtype):
    """Attempt to determine the min and max values of a datatype.

    :arg dtype: A numpy datatype.
    :returns: a 2-tuple of min, max
    :raises ValueError: If numeric limits could not be determined.
    """
    try:
        info = numpy.finfo(dtype)
    except ValueError:
        # maybe an int?
        try:
            info = numpy.iinfo(dtype)
        except ValueError as e:
            raise ValueError("Unable to determine numeric limits from %s" % dtype) from e
    return DTypeLimit(info.min, info.max)


class OpaqueType(lp.types.OpaqueType):
    def __init__(self, name):
        super().__init__(name=name)

    def __repr__(self):
        return self.name
