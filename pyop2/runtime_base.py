# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""Base classes for OP2 objects. The versions here extend those from the
:mod:`base` module to include runtime data information which is backend
independent. Individual runtime backends should subclass these as
required to implement backend-specific features.

.. _MatMPIAIJSetPreallocation: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html
"""

import numpy as np
import operator

from exceptions import *
from utils import *
import configuration as cfg
import base
from base import READ, WRITE, RW, INC, MIN, MAX, IterationSpace
from base import DataCarrier, IterationIndex, i, IdentityMap, Kernel, Global
from base import _parloop_cache, _empty_parloop_cache, _parloop_cache_size
import op_lib_core as core
from mpi4py import MPI
from petsc4py import PETSc

PYOP2_COMM = None

def get_mpi_communicator():
    """The MPI Communicator used by PyOP2."""
    global PYOP2_COMM
    return PYOP2_COMM

def set_mpi_communicator(comm):
    """Set the MPI communicator for parallel communication."""
    global PYOP2_COMM
    if comm is None:
        PYOP2_COMM = MPI.COMM_WORLD
    elif type(comm) is int:
        # If it's come from Fluidity where an MPI_Comm is just an
        # integer.
        PYOP2_COMM = MPI.Comm.f2py(comm)
    else:
        PYOP2_COMM = comm
    # PETSc objects also need to be built on the same communicator.
    PETSc.Sys.setDefaultComm(PYOP2_COMM)

# Data API

class Arg(base.Arg):
    """An argument to a :func:`par_loop`.

    .. warning:: User code should not directly instantiate :class:`Arg`. Instead, use the call syntax on the :class:`DataCarrier`.
    """

    def halo_exchange_begin(self):
        """Begin halo exchange for the argument if a halo update is required.
        Doing halo exchanges only makes sense for :class:`Dat` objects."""
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        assert not self._in_flight, \
            "Halo exchange already in flight for Arg %s" % self
        if self.access in [READ, RW] and self.data.needs_halo_update:
            self.data.needs_halo_update = False
            self._in_flight = True
            self.data.halo_exchange_begin()

    def halo_exchange_end(self):
        """End halo exchange if it is in flight.
        Doing halo exchanges only makes sense for :class:`Dat` objects."""
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        if self.access in [READ, RW] and self._in_flight:
            self._in_flight = False
            self.data.halo_exchange_end()

    def reduction_begin(self):
        """Begin reduction for the argument if its access is INC, MIN, or MAX.
        Doing a reduction only makes sense for :class:`Global` objects."""
        assert self._is_global, \
            "Doing global reduction only makes sense for Globals"
        assert not self._in_flight, \
            "Reduction already in flight for Arg %s" % self
        if self.access is not READ:
            self._in_flight = True
            if self.access is INC:
                op = MPI.SUM
            elif self.access is MIN:
                op = MPI.MIN
            elif self.access is MAX:
                op = MPI.MAX
            # If the MPI supports MPI-3, this could be MPI_Iallreduce
            # instead, to allow overlapping comp and comms.
            # We must reduce into a temporary buffer so that when
            # executing over the halo region, which occurs after we've
            # called this reduction, we don't subsequently overwrite
            # the result.
            PYOP2_COMM.Allreduce(self.data._data, self.data._buf, op=op)

    def reduction_end(self):
        """End reduction for the argument if it is in flight.
        Doing a reduction only makes sense for :class:`Global` objects."""
        assert self._is_global, \
            "Doing global reduction only makes sense for Globals"
        if self.access is not READ and self._in_flight:
            self._in_flight = False
            # Must have a copy here, because otherwise we just grab a
            # pointer.
            self.data._data = np.copy(self.data._buf)

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_arg(self)
        return self._lib_handle

class Set(base.Set):
    """OP2 set."""

    @validate_type(('size', (int, tuple, list), SizeTypeError))
    def __init__(self, size, name=None, halo=None):
        base.Set.__init__(self, size, name, halo)

    @classmethod
    def fromhdf5(cls, f, name):
        """Construct a :class:`Set` from set named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        size = slot.value.astype(np.int)
        shape = slot.shape
        if shape != (1,):
            raise SizeTypeError("Shape of %s is incorrect" % name)
        return cls(size[0], name)

    def __call__(self, *dims):
        return IterationSpace(self, dims)

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_set(self)
        return self._lib_handle

class Halo(base.Halo):
    def __init__(self, sends, receives, comm=PYOP2_COMM, gnn2unn=None):
        base.Halo.__init__(self, sends, receives, gnn2unn)
        if type(comm) is int:
            self._comm = MPI.Comm.f2py(comm)
        else:
            self._comm = comm
        # FIXME: is this a necessity?
        assert self._comm == PYOP2_COMM, "Halo communicator not PYOP2_COMM"
        rank = self._comm.rank
        size = self._comm.size

        assert len(self._sends) == size, \
            "Invalid number of sends for Halo, got %d, wanted %d" % \
            (len(self._sends), size)
        assert len(self._receives) == size, \
            "Invalid number of receives for Halo, got %d, wanted %d" % \
            (len(self._receives), size)

        assert self._sends[rank].size == 0, \
            "Halo was specified with self-sends on rank %d" % rank
        assert self._receives[rank].size == 0, \
            "Halo was specified with self-receives on rank %d" % rank

    @property
    def comm(self):
        """The MPI communicator this :class:`Halo`'s communications
    should take place over"""
        return self._comm

    def verify(self, s):
        """Verify that this :class:`Halo` is valid for a given
:class:`Set`."""
        for dest, sends in enumerate(self.sends):
            assert (sends >= 0).all() and (sends < s.size).all(), \
                "Halo send to %d is invalid (outside owned elements)" % dest

        for source, receives in enumerate(self.receives):
            assert (receives >= s.size).all() and \
                (receives < s.total_size).all(), \
                "Halo receive from %d is invalid (not in halo elements)" % \
                source

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['_comm']
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        # FIXME: This will break for custom halo communicators
        self._comm = PYOP2_COMM

class Dat(base.Dat):
    """OP2 vector data. A ``Dat`` holds a value for every member of a :class:`Set`."""

    def __init__(self, dataset, dim, data=None, dtype=None, name=None,
                 soa=None, uid=None):
        base.Dat.__init__(self, dataset, dim, data, dtype, name, soa, uid)
        halo = dataset.halo
        if halo is not None:
            self._send_reqs = [None]*halo.comm.size
            self._send_buf = [None]*halo.comm.size
            self._recv_reqs = [None]*halo.comm.size
            self._recv_buf = [None]*halo.comm.size

    def _check_shape(self, other):
        pass

    def _op(self, other, op):
        if np.isscalar(other):
            return Dat(self.dataset, self.dim,
                       op(self._data, as_type(other, self.dtype)), self.dtype)
        self._check_shape(other)
        return Dat(self.dataset, self.dim,
                   op(self._data, as_type(other.data, self.dtype)), self.dtype)

    def _iop(self, other, op):
        if np.isscalar(other):
            op(self._data, as_type(other, self.dtype))
        else:
            self._check_shape(other)
            op(self._data, as_type(other.data, self.dtype))
        return self

    def __add__(self, other):
        """Pointwise addition of fields."""
        return self._op(other, operator.add)

    def __sub__(self, other):
        """Pointwise subtraction of fields."""
        return self._op(other, operator.sub)

    def __mul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._op(other, operator.mul)

    def __div__(self, other):
        """Pointwise division or scaling of fields."""
        return self._op(other, operator.div)

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        return self._iop(other, operator.iadd)

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        return self._iop(other, operator.isub)

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        return self._iop(other, operator.imul)

    def __idiv__(self, other):
        """Pointwise division or scaling of fields."""
        return self._iop(other, operator.idiv)

    def halo_exchange_begin(self):
        """Begin halo exchange."""
        halo = self.dataset.halo
        if halo is None:
            return
        for dest,ele in enumerate(halo.sends):
            if ele.size == 0:
                # Don't send to self (we've asserted that ele.size ==
                # 0 previously) or if there are no elements to send
                self._send_reqs[dest] = MPI.REQUEST_NULL
                continue
            self._send_buf[dest] = self._data[ele]
            self._send_reqs[dest] = halo.comm.Isend(self._send_buf[dest],
                                                    dest=dest, tag=self._id)
        for source,ele in enumerate(halo.receives):
            if ele.size == 0:
                # Don't receive from self or if there are no elements
                # to receive
                self._recv_reqs[source] = MPI.REQUEST_NULL
                continue
            self._recv_buf[source] = self._data[ele]
            self._recv_reqs[source] = halo.comm.Irecv(self._recv_buf[source],
                                                      source=source, tag=self._id)

    def halo_exchange_end(self):
        """End halo exchange. Waits on MPI recv."""
        halo = self.dataset.halo
        if halo is None:
            return
        MPI.Request.Waitall(self._recv_reqs)
        MPI.Request.Waitall(self._send_reqs)
        self._send_buf = [None]*len(self._send_buf)
        for source, buf in enumerate(self._recv_buf):
            if buf is not None:
                self._data[halo.receives[source]] = buf
        self._recv_buf = [None]*len(self._recv_buf)

    @property
    def norm(self):
        """The L2-norm on the flattened vector."""
        return np.linalg.norm(self._data)

    @classmethod
    def fromhdf5(cls, dataset, f, name):
        """Construct a :class:`Dat` from a Dat named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        data = slot.value
        dim = slot.shape[1:]
        soa = slot.attrs['type'].find(':soa') > 0
        if len(dim) < 1:
            raise DimTypeError("Invalid dimension value %s" % dim)
        ret = cls(dataset, dim, data, name=name, soa=soa)
        return ret

    @property
    def vec(self):
        """PETSc Vec appropriate for this Dat."""
        if not hasattr(self, '_vec'):
            size = (self.dataset.size * self.cdim, None)
            self._vec = PETSc.Vec().createWithArray(self._data, size=size)
        return self._vec

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_dat(self)
        return self._lib_handle

class Const(base.Const):
    """Data that is constant for any element of any set."""

    @classmethod
    def fromhdf5(cls, f, name):
        """Construct a :class:`Const` from const named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        dim = slot.shape
        data = slot.value
        if len(dim) < 1:
            raise DimTypeError("Invalid dimension value %s" % dim)
        return cls(dim, data, name)

class Map(base.Map):
    """OP2 map, a relation between two :class:`Set` objects."""

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_map(self)
        return self._lib_handle

    @classmethod
    def fromhdf5(cls, iterset, dataset, f, name):
        """Construct a :class:`Map` from set named ``name`` in HDF5 data ``f``"""
        slot = f[name]
        values = slot.value
        dim = slot.shape[1:]
        if len(dim) != 1:
            raise DimTypeError("Unrecognised dimension value %s" % dim)
        return cls(iterset, dataset, dim[0], values, name)

_sparsity_cache = dict()
def _empty_sparsity_cache():
    _sparsity_cache.clear()

class Sparsity(base.Sparsity):
    """OP2 Sparsity, a matrix structure derived from the union of the outer product of pairs of :class:`Map` objects."""

    @validate_type(('maps', (Map, tuple), MapTypeError), \
                   ('dims', (int, tuple), TypeError))
    def __new__(cls, maps, dims, name=None):
        key = (maps, as_tuple(dims, int, 2))
        cached = _sparsity_cache.get(key)
        if cached is not None:
            return cached
        return super(Sparsity, cls).__new__(cls, maps, dims, name)

    @validate_type(('maps', (Map, tuple), MapTypeError), \
                   ('dims', (int, tuple), TypeError))
    def __init__(self, maps, dims, name=None):
        if getattr(self, '_cached', False):
            return
        for m in maps:
            for n in as_tuple(m, Map):
                if len(n.values) == 0:
                    raise MapValueError("Unpopulated map values when trying to build sparsity.")
        super(Sparsity, self).__init__(maps, dims, name)
        key = (maps, as_tuple(dims, int, 2))
        self._cached = True
        core.build_sparsity(self, parallel=PYOP2_COMM.size > 1)
        _sparsity_cache[key] = self

    def __del__(self):
        core.free_sparsity(self)

    @property
    def rowptr(self):
        """Row pointer array of CSR data structure."""
        return self._rowptr

    @property
    def colidx(self):
        """Column indices array of CSR data structure."""
        return self._colidx

    @property
    def nnz(self):
        """Array containing the number of non-zeroes in the various rows of the
        diagonal portion of the local submatrix.

        This is the same as the parameter `d_nnz` used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return self._d_nnz

    @property
    def onnz(self):
        """Array containing the number of non-zeroes in the various rows of the
        off-diagonal portion of the local submatrix.

        This is the same as the parameter `o_nnz` used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return self._o_nnz

    @property
    def nz(self):
        """Number of non-zeroes per row in diagonal portion of the local
        submatrix.

        This is the same as the parameter `d_nz` used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return int(self._d_nz)

    @property
    def onz(self):
        """Number of non-zeroes per row in off-diagonal portion of the local
        submatrix.

        This is the same as the parameter o_nz used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return int(self._o_nz)

class Mat(base.Mat):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    def __init__(self, *args, **kwargs):
        super(Mat, self).__init__(*args, **kwargs)
        self._handle = None

    def _init(self):
        if not self.dtype == PETSc.ScalarType:
            raise RuntimeError("Can only create a matrix of type %s, %s is not supported" \
                    % (PETSc.ScalarType, self.dtype))
        if PYOP2_COMM.size == 1:
            mat = PETSc.Mat()
            row_lg = PETSc.LGMap()
            col_lg = PETSc.LGMap()
            rdim, cdim = self.sparsity.dims
            row_lg.create(indices=np.arange(self.sparsity.nrows * rdim, dtype=PETSc.IntType))
            col_lg.create(indices=np.arange(self.sparsity.ncols * cdim, dtype=PETSc.IntType))
            self._array = np.zeros(self.sparsity.nz, dtype=PETSc.RealType)
            # We're not currently building a blocked matrix, so need to scale the
            # number of rows and columns by the sparsity dimensions
            # FIXME: This needs to change if we want to do blocked sparse
            # NOTE: using _rowptr and _colidx since we always want the host values
            mat.createAIJWithArrays((self.sparsity.nrows*rdim, self.sparsity.ncols*cdim),
                                    (self.sparsity._rowptr, self.sparsity._colidx, self._array))
            mat.setLGMap(rmap=row_lg, cmap=col_lg)
        else:
            mat = PETSc.Mat()
            row_lg = PETSc.LGMap()
            col_lg = PETSc.LGMap()
            # FIXME: probably not right for vector fields
            row_lg.create(indices=self.sparsity.maps[0][0].dataset.halo.global_to_petsc_numbering)
            col_lg.create(indices=self.sparsity.maps[0][1].dataset.halo.global_to_petsc_numbering)
            rdim, cdim = self.sparsity.dims
            mat.createAIJ(size=((self.sparsity.nrows*rdim, None),
                                (self.sparsity.ncols*cdim, None)),
                          nnz=(self.sparsity.nnz, self.sparsity.onnz))
            mat.setLGMap(rmap=row_lg, cmap=col_lg)
            mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, True)
            mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)
            mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        self._handle = mat

    def zero(self):
        """Zero the matrix."""
        self.handle.zeroEntries()

    def zero_rows(self, rows, diag_val):
        """Zeroes the specified rows of the matrix, with the exception of the
        diagonal entry, which is set to diag_val. May be used for applying
        strong boundary conditions."""
        self.handle.zeroRowsLocal(rows, diag_val)

    def _assemble(self):
        self.handle.assemble()

    @property
    def array(self):
        """Array of non-zero values."""
        if not hasattr(self, '_array'):
            self._init()
        return self._array

    @property
    def values(self):
        return self.handle[:,:]

    @property
    def handle(self):
        """Petsc4py Mat holding matrix data."""
        if self._handle is None:
            self._init()
        return self._handle

class ParLoop(base.ParLoop):
    def compute(self):
        """Executes the kernel over all members of the iteration space."""
        raise RuntimeError('Must select a backend')

# FIXME: Eventually (when we have a proper OpenCL solver) this wants to go in
# sequential
class Solver(base.Solver, PETSc.KSP):

    _cnt = 0

    def __init__(self, parameters=None, **kwargs):
        super(Solver, self).__init__(parameters, **kwargs)
        self.create(PETSc.COMM_WORLD)
        converged_reason = self.ConvergedReason()
        self._reasons = dict([(getattr(converged_reason,r), r) \
                              for r in dir(converged_reason) \
                              if not r.startswith('_')])

    def _set_parameters(self):
        self.setType(self.parameters['linear_solver'])
        self.getPC().setType(self.parameters['preconditioner'])
        self.rtol = self.parameters['relative_tolerance']
        self.atol = self.parameters['absolute_tolerance']
        self.divtol = self.parameters['divergence_tolerance']
        self.max_it = self.parameters['maximum_iterations']
        if self.parameters['plot_convergence']:
            self.parameters['monitor_convergence'] = True

    def solve(self, A, x, b):
        self._set_parameters()
        self.setOperators(A.handle)
        self.setFromOptions()
        if self.parameters['monitor_convergence']:
            self.reshist = []
            def monitor(ksp, its, norm):
                self.reshist.append(norm)
                print "%3d KSP Residual norm %14.12e" % (its, norm)
            self.setMonitor(monitor)
        # Not using super here since the MRO would call base.Solver.solve
        PETSc.KSP.solve(self, b.vec, x.vec)
        x.needs_halo_update = True
        if self.parameters['monitor_convergence']:
            self.cancelMonitor()
            if self.parameters['plot_convergence']:
                try:
                    import pylab
                    pylab.semilogy(self.reshist)
                    pylab.title('Convergence history')
                    pylab.xlabel('Iteration')
                    pylab.ylabel('Residual norm')
                    pylab.savefig('%sreshist_%04d.png' % (self.parameters['plot_prefix'], Solver._cnt))
                    Solver._cnt += 1
                except ImportError:
                    from warnings import warn
                    warn("pylab not available, not plotting convergence history.")
        r = self.getConvergedReason()
        if cfg.debug:
            print "Converged reason: %s" % self._reasons[r]
            print "Iterations: %s" % self.getIterationNumber()
            print "Residual norm: %s" % self.getResidualNorm()
        if r < 0:
            msg = "KSP Solver failed to converge in %d iterations: %s (Residual norm: %e)" \
                    % (self.getIterationNumber(), self._reasons[r], self.getResidualNorm())
            if self.parameters['error_on_nonconvergence']:
                raise RuntimeError(msg)
            else:
                from warnings import warn
                warn(msg)
