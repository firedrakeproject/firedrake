from __future__ import absolute_import, print_function, division
from pyop2.utils import maybe_setflags
from mpi4py import MPI

import firedrake.dmplex as dmplex


_MPI_types = {}


def _get_mtype(dat):
    """Get an MPI datatype corresponding to a Dat.

    This builds (if necessary a contiguous derived datatype of the
    correct size)."""
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
        else:
            typ = btype.Create_contiguous(dat.cdim)
            typ.Commit()
        _MPI_types[key] = typ
        return typ


class Halo(object):
    """Build a Halo for a function space.

    :arg dm:  The DM describing the data layout (has a Section attached).

    The halo is implemented using a PETSc SF (star forest) object and
    is usable as a PyOP2 :class:`pyop2.Halo`."""

    def __init__(self, dm):
        lsec = dm.getDefaultSection()
        gsec = dm.getDefaultGlobalSection()
        dm.createDefaultSF(lsec, gsec)
        sf = dm.getDefaultSF()

        # The full SF is designed for GlobalToLocal or LocalToGlobal
        # where the input and output buffers are different.  So on the
        # local rank, it copies data from input to output.  However,
        # our halo exchanges use the same buffer for input and output
        # (so we don't need to do the local copy).  To facilitate
        # this, prune the SF to remove all the roots that reference
        # the local rank.
        self.sf = dmplex.prune_sf(sf)
        self.comm = self.sf.comm.tompi4py()
        self.sf.setFromOptions()
        if self.sf.getType() != self.sf.Type.BASIC:
            raise RuntimeError("Windowed SFs expose bugs in OpenMPI (use -sf_type basic)")
        if self.comm.size == 1:
            self._gnn2unn = None
        self._gnn2unn = dmplex.make_global_numbering(lsec, gsec)

    def begin(self, dat, reverse=False):
        """Begin a halo exchange.

        :arg dat: The :class:`pyop2.Dat` to start a halo exchange on.
        :arg reverse: (optional) perform a reverse halo exchange.

        .. note::

           If ``reverse`` is ``True`` then the input buffer
           may not be touched before calling :meth:`.end`."""
        if self.comm.size == 1:
            return
        mtype = _get_mtype(dat)
        dmplex.halo_begin(self.sf, dat, mtype, reverse)

    def end(self, dat, reverse=False):
        """End a halo exchange.

        :arg dat: The :class:`pyop2.Dat` to end a halo exchange on.
        :arg reverse: (optional) perform a reverse halo exchange.

        See also :meth:`.begin`."""
        if self.comm.size == 1:
            return
        mtype = _get_mtype(dat)
        maybe_setflags(dat._data, write=True)
        dmplex.halo_end(self.sf, dat, mtype, reverse)
        maybe_setflags(dat._data, write=False)

    def verify(self, *args):
        """No-op"""
        pass

    @property
    def global_to_petsc_numbering(self):
        """Return a mapping from global (process-local) to universal
    (process-global) numbers"""
        return self._gnn2unn
