import collections

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

ScalarType = np.dtype(PETSc.ScalarType)
IntType = np.dtype(PETSc.IntType)
PointerType = np.uintp


# dtypes can either be a numpy dtype object or just a class deriving from np.number
# return isinstance(obj, np.dtype) or issubclass(obj, np.number)
DTypeT = np.dtype | type


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


_numpy_types = {}


def as_numpy_dtype(mpi_dtype):
    """Return the numpy datatype corresponding to the MPI datatype.

    This only works for contiguous datatypes.

    """
    try:
        # possibly unsafe if handles are recycled, but OK, because we
        # hold on to the contig types
        return _numpy_types[mpi_dtype.py2f()]
    except KeyError:
        base, combiner, _ = mpi_dtype.decode()
        while combiner == "DUP":
            base, combiner, _ = base.decode()
        if combiner != "CONTIGUOUS":
            raise RuntimeError("Can only handle contiguous types")
        try:
            tdict = MPI.__TypeDict__
        except AttributeError:
            tdict = MPI._typedict

        tdict = dict((v.py2f(), k) for k, v in tdict.items())
        try:
            base = tdict[base.py2f()]
        except KeyError:
            raise RuntimeError("Unhandled base datatype %r", base)
        return _numpy_types.setdefault(mpi_dtype.py2f(), base)


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
            raise ValueError(
                "Unable to determine numeric limits from %s" % dtype
            ) from e
    return DTypeLimit(info.min, info.max)
