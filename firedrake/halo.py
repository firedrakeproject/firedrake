from pyop2 import mpi, op2, utils
from mpi4py import MPI
import numpy
from functools import partial

from firedrake.petsc import PETSc
import firedrake.cython.dmcommon as dmcommon


_MPI_types = {}


def _get_mtype(dat):
    """Get an MPI datatype corresponding to a Dat.

    This builds (if necessary a contiguous derived datatype of the
    correct size).

    Also returns if it is a builtin type.
    """
    key = (dat.dtype, dat.cdim)
    try:
        return _MPI_types[key]
    except KeyError:
        try:
            tdict = MPI.__TypeDict__
        except AttributeError:
            tdict = MPI._typedict
        try:
            btype = tdict[dat.dtype.char]
        except KeyError:
            raise RuntimeError("Unknown base type %r", dat.dtype)
        if dat.cdim == 1:
            typ = btype
            builtin = True
        else:
            typ = btype.Create_contiguous(dat.cdim)
            typ.Commit()
            builtin = False
        return _MPI_types.setdefault(key, (typ, builtin))


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
        return _numpy_types.setdefault(datatype.py2f(), base)


def reduction_op(op, invec, inoutvec, datatype):
    dtype = _get_dtype(datatype)
    invec = numpy.frombuffer(invec, dtype=dtype)
    inoutvec = numpy.frombuffer(inoutvec, dtype=dtype)
    inoutvec[:] = op(invec, inoutvec)


_contig_min_op = MPI.Op.Create(partial(reduction_op, numpy.minimum), commute=True)
_contig_max_op = MPI.Op.Create(partial(reduction_op, numpy.maximum), commute=True)


class Halo(op2.Halo):
    """Build a Halo for a function space.

    :arg dm: The DM describing the topology.
    :arg section: The data layout.

    The halo is implemented using a PETSc SF (star forest) object and
    is usable as a PyOP2 :class:`pyop2.types.halo.Halo` ."""

    def __init__(self, dm, section, comm):
        super(Halo, self).__init__()
        self.comm = comm
        self._comm = mpi.internal_comm(comm, self)
        # Use a DM to create the halo SFs
        if MPI.Comm.Compare(comm, dm.comm.tompi4py()) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Communicator used to create `Halo` must be at least congruent to the communicator used to create the mesh")
        self.dm = PETSc.DMShell().create(self._comm)
        self.dm.setPointSF(dm.getPointSF())
        self.dm.setDefaultSection(section)

    @utils.cached_property
    def sf(self):
        sf = dmcommon.create_halo_exchange_sf(self.dm)
        sf.setFromOptions()
        if sf.getType() != sf.Type.BASIC:
            raise RuntimeError("Windowed SFs expose bugs in OpenMPI (use -sf_type basic)")
        return sf

    @utils.cached_property
    def comm(self):
        return self.comm

    @utils.cached_property
    def local_to_global_numbering(self):
        lsec = self.dm.getDefaultSection()
        gsec = self.dm.getDefaultGlobalSection()
        return dmcommon.make_global_numbering(lsec, gsec)

    @PETSc.Log.EventDecorator()
    def global_to_local_begin(self, dat, insert_mode):
        assert insert_mode is op2.WRITE, "Only WRITE GtoL supported"
        if self.comm.size == 1:
            return
        mtype, _ = _get_mtype(dat)
        self.sf.bcastBegin(mtype, dat._data, dat._data, MPI.REPLACE)

    @PETSc.Log.EventDecorator()
    def global_to_local_end(self, dat, insert_mode):
        assert insert_mode is op2.WRITE, "Only WRITE GtoL supported"
        if self.comm.size == 1:
            return
        mtype, _ = _get_mtype(dat)
        self.sf.bcastEnd(mtype, dat._data, dat._data, MPI.REPLACE)

    @PETSc.Log.EventDecorator()
    def local_to_global_begin(self, dat, insert_mode):
        assert insert_mode in {op2.INC, op2.MIN, op2.MAX}, "%s LtoG not supported" % insert_mode
        if self.comm.size == 1:
            return
        mtype, builtin = _get_mtype(dat)
        op = {(False, op2.INC): MPI.SUM,
              (True, op2.INC): MPI.SUM,
              (False, op2.MIN): _contig_min_op,
              (True, op2.MIN): MPI.MIN,
              (False, op2.MAX): _contig_max_op,
              (True, op2.MAX): MPI.MAX}[(builtin, insert_mode)]
        self.sf.reduceBegin(mtype, dat._data, dat._data, op)

    @PETSc.Log.EventDecorator()
    def local_to_global_end(self, dat, insert_mode):
        assert insert_mode in {op2.INC, op2.MIN, op2.MAX}, "%s LtoG not supported" % insert_mode
        if self.comm.size == 1:
            return
        mtype, builtin = _get_mtype(dat)
        op = {(False, op2.INC): MPI.SUM,
              (True, op2.INC): MPI.SUM,
              (False, op2.MIN): _contig_min_op,
              (True, op2.MIN): MPI.MIN,
              (False, op2.MAX): _contig_max_op,
              (True, op2.MAX): MPI.MAX}[(builtin, insert_mode)]
        self.sf.reduceEnd(mtype, dat._data, dat._data, op)
