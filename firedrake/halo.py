from pyop2 import op2
from pyop2 import utils
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
    is usable as a PyOP2 :class:`pyop2.Halo`."""

    def __init__(self, dm, section):
        super(Halo, self).__init__()
        # Use a DM to create the halo SFs
        self.dm = PETSc.DMShell().create(dm.comm)
        self.dm.setPointSF(dm.getPointSF())
        self.dm.setDefaultSection(section)

    @utils.cached_property
    def sf(self):
        lsec = self.dm.getDefaultSection()
        gsec = self.dm.getDefaultGlobalSection()
        self.dm.createDefaultSF(lsec, gsec)
        # The full SF is designed for GlobalToLocal or LocalToGlobal
        # where the input and output buffers are different.  So on the
        # local rank, it copies data from input to output.  However,
        # our halo exchanges use the same buffer for input and output
        # (so we don't need to do the local copy).  To facilitate
        # this, prune the SF to remove all the roots that reference
        # the local rank.
        sf = dmcommon.prune_sf(self.dm.getDefaultSF())
        sf.setFromOptions()
        if sf.getType() != sf.Type.BASIC:
            raise RuntimeError("Windowed SFs expose bugs in OpenMPI (use -sf_type basic)")
        return sf

    @utils.cached_property
    def comm(self):
        return self.dm.comm.tompi4py()

    @utils.cached_property
    def local_to_global_numbering(self):
        lsec = self.dm.getDefaultSection()
        gsec = self.dm.getDefaultGlobalSection()
        return dmcommon.make_global_numbering(lsec, gsec)

    def global_to_local_begin(self, dat, insert_mode):
        assert insert_mode is op2.WRITE, "Only WRITE GtoL supported"
        if self.comm.size == 1:
            return
        mtype, _ = _get_mtype(dat)
        dmcommon.halo_begin(self.sf, dat, mtype, False)

    def global_to_local_end(self, dat, insert_mode):
        assert insert_mode is op2.WRITE, "Only WRITE GtoL supported"
        if self.comm.size == 1:
            return
        mtype, _ = _get_mtype(dat)
        dmcommon.halo_end(self.sf, dat, mtype, False)

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
        dmcommon.halo_begin(self.sf, dat, mtype, True, op=op)

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
        dmcommon.halo_end(self.sf, dat, mtype, True, op=op)
