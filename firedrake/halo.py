from pyop2 import op2
from pyop2.utils import maybe_setflags
from mpi4py import MPI

import dmplex


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
            btype = MPI.__TypeDict__[dat.dtype.char]
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

    :arg fs:  The :class:`.FunctionSpace` to build a :class:`Halo` for.

    The halo is implemented using a PETSc SF (star forest) object and
    is usable as a PyOP2 :class:`pyop2.Halo`."""

    def __init__(self, fs):
        dm = fs.mesh()._plex
        local_section = fs._global_numbering
        global_section = fs._universal_numbering
        self.sf = dmplex.make_sf(dm, local_section, global_section)
        if op2.MPI.comm.size == 1:
            self._gnn2unn = None
        self._gnn2unn = dmplex.make_global_numbering(local_section,
                                                     global_section)

    @property
    def comm(self):
        """The communicator for this halo."""
        return self.sf.comm

    def begin(self, dat, reverse=False):
        """Begin a halo exchange.

        :arg dat: The :class:`pyop2.Dat` to start a halo exchange on.
        :arg reverse: (optional) perform a reverse halo exchange.

        .. note::

           If :data:`reverse` is :data:`True` then the input buffer
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
