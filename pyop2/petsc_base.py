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

from contextlib import contextmanager
from petsc4py import PETSc, __version__ as petsc4py_version

import base
from base import *
from backends import _make_object
from logger import debug, warning
from versioning import CopyOnWrite
import mpi
from mpi import collective

if petsc4py_version < '3.4':
    raise RuntimeError("Incompatible petsc4py version %s. At least version 3.4 is required."
                       % petsc4py_version)


class MPIConfig(mpi.MPIConfig):

    def __init__(self):
        super(MPIConfig, self).__init__()
        PETSc.Sys.setDefaultComm(self.comm)

    @mpi.MPIConfig.comm.setter
    @collective
    def comm(self, comm):
        """Set the MPI communicator for parallel communication."""
        self.COMM = mpi._check_comm(comm)
        # PETSc objects also need to be built on the same communicator.
        PETSc.Sys.setDefaultComm(self.comm)

MPI = MPIConfig()
# Override MPI configuration
mpi.MPI = MPI


class Dat(base.Dat):

    @contextmanager
    def vec_context(self, readonly=True):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Dat`.

        :param readonly: Access the data read-only (use :meth:`Dat.data_ro`)
                         or read-write (use :meth:`Dat.data`). Read-write
                         access requires a halo update."""

        acc = (lambda d: d.data_ro) if readonly else (lambda d: d.data)
        # Getting the Vec needs to ensure we've done all current computation.
        self._force_evaluation()
        if not hasattr(self, '_vec'):
            size = (self.dataset.size * self.cdim, None)
            self._vec = PETSc.Vec().createWithArray(acc(self), size=size)
        yield self._vec
        if not readonly:
            self.needs_halo_update = True

    @property
    @collective
    def vec(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're allowed to modify the data you get back from this view."""
        return self.vec_context(readonly=False)

    @property
    @collective
    def vec_ro(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're not allowed to modify the data you get back from this view."""
        return self.vec_context()

    @collective
    def dump(self, filename):
        """Dump the vector to file ``filename`` in PETSc binary format."""
        base._trace.evaluate(set([self]), set())
        vwr = PETSc.Viewer().createBinary(filename, PETSc.Viewer.Mode.WRITE)
        self.vec.view(vwr)


class MixedDat(base.MixedDat):

    @contextmanager
    def vecscatter(self, readonly=True):
        """A context manager scattering the arrays of all components of this
        :class:`MixedDat` into a contiguous :class:`PETSc.Vec` and reverse
        scattering to the original arrays when exiting the context.

        :param readonly: Access the data read-only (use :meth:`Dat.data_ro`)
                         or read-write (use :meth:`Dat.data`). Read-write
                         access requires a halo update.

        .. note::

           The :class:`~PETSc.Vec` obtained from this context is in
           the correct order to be left multiplied by a compatible
           :class:`MixedMat`.  In parallel it is *not* just a
           concatenation of the underlying :class:`Dat`\s."""

        acc = (lambda d: d.vec_ro) if readonly else (lambda d: d.vec)
        # Allocate memory for the contiguous vector, create the scatter
        # contexts and stash them on the object for later reuse
        if not (hasattr(self, '_vec') and hasattr(self, '_sctxs')):
            self._vec = PETSc.Vec().create()
            # Size of flattened vector is product of size and cdim of each dat
            sz = sum(d.dataset.size * d.dataset.cdim for d in self._dats)
            self._vec.setSizes((sz, None))
            self._vec.setUp()
            self._sctxs = []
            # To be compatible with a MatNest (from a MixedMat) the
            # ordering of a MixedDat constructed of Dats (x_0, ..., x_k)
            # on P processes is:
            # (x_0_0, x_1_0, ..., x_k_0, x_0_1, x_1_1, ..., x_k_1, ..., x_k_P)
            # That is, all the Dats from rank 0, followed by those of
            # rank 1, ...
            # Hence the offset into the global Vec is the exclusive
            # prefix sum of the local size of the mixed dat.
            offset = MPI.comm.exscan(sz)
            if offset is None:
                offset = 0

            for d in self._dats:
                sz = d.dataset.size * d.dataset.cdim
                with acc(d) as v:
                    vscat = PETSc.Scatter().create(v, None, self._vec,
                                                   PETSc.IS().createStride(sz, offset, 1))
                offset += sz
                self._sctxs.append(vscat)
        # Do the actual forward scatter to fill the full vector with values
        for d, vscat in zip(self._dats, self._sctxs):
            with acc(d) as v:
                vscat.scatterBegin(v, self._vec, addv=PETSc.InsertMode.INSERT_VALUES)
                vscat.scatterEnd(v, self._vec, addv=PETSc.InsertMode.INSERT_VALUES)
        yield self._vec
        if not readonly:
            # Reverse scatter to get the values back to their original locations
            for d, vscat in zip(self._dats, self._sctxs):
                with acc(d) as v:
                    vscat.scatterBegin(self._vec, v, addv=PETSc.InsertMode.INSERT_VALUES,
                                       mode=PETSc.ScatterMode.REVERSE)
                    vscat.scatterEnd(self._vec, v, addv=PETSc.InsertMode.INSERT_VALUES,
                                     mode=PETSc.ScatterMode.REVERSE)
            self.needs_halo_update = True

    @property
    @collective
    def vec(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're allowed to modify the data you get back from this view."""
        return self.vecscatter(readonly=False)

    @property
    @collective
    def vec_ro(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're not allowed to modify the data you get back from this view."""
        return self.vecscatter()


class Mat(base.Mat, CopyOnWrite):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    @collective
    def _init(self):
        if not self.dtype == PETSc.ScalarType:
            raise RuntimeError("Can only create a matrix of type %s, %s is not supported"
                               % (PETSc.ScalarType, self.dtype))
        # If the Sparsity is defined on MixedDataSets, we need to build a MatNest
        if self.sparsity.shape > (1, 1):
            self._init_nest()
        else:
            self._init_block()
        self._ever_assembled = False

    def _init_nest(self):
        mat = PETSc.Mat()
        self._blocks = []
        rows, cols = self.sparsity.shape
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(Mat(self.sparsity[i, j], self.dtype,
                           '_'.join([self.name, str(i), str(j)])))
            self._blocks.append(row)
        # PETSc Mat.createNest wants a flattened list of Mats
        mat.createNest([[m.handle for m in row_] for row_ in self._blocks])
        self._handle = mat

    def _init_block(self):
        self._blocks = [[self]]
        mat = PETSc.Mat()
        row_lg = PETSc.LGMap()
        col_lg = PETSc.LGMap()
        rdim, cdim = self.sparsity.dims
        if MPI.comm.size == 1:
            # The PETSc local to global mapping is the identity in the sequential case
            row_lg.create(
                indices=np.arange(self.sparsity.nrows * rdim, dtype=PETSc.IntType))
            col_lg.create(
                indices=np.arange(self.sparsity.ncols * cdim, dtype=PETSc.IntType))
            self._array = np.zeros(self.sparsity.nz, dtype=PETSc.RealType)
            # We're not currently building a blocked matrix, so need to scale the
            # number of rows and columns by the sparsity dimensions
            # FIXME: This needs to change if we want to do blocked sparse
            # NOTE: using _rowptr and _colidx since we always want the host values
            mat.createAIJWithArrays(
                (self.sparsity.nrows * rdim, self.sparsity.ncols * cdim),
                (self.sparsity._rowptr, self.sparsity._colidx, self._array))
        else:
            # FIXME: probably not right for vector fields
            # We get the PETSc local to global mapping from the halo
            row_lg.create(indices=self.sparsity.rmaps[
                          0].toset.halo.global_to_petsc_numbering)
            col_lg.create(indices=self.sparsity.cmaps[
                          0].toset.halo.global_to_petsc_numbering)
            # PETSc has utility for turning a local to global map into
            # a blocked one and vice versa, if rdim or cdim are > 1,
            # the global_to_petsc_numbering we have is a blocked map,
            # however, we can't currently generate the correct code
            # for that case, so build the unblocked map and use that.
            # This is a temporary fix until we do things properly.
            row_lg = row_lg.unblock(rdim)
            col_lg = col_lg.unblock(cdim)

            mat.createAIJ(size=((self.sparsity.nrows * rdim, None),
                                (self.sparsity.ncols * cdim, None)),
                          nnz=(self.sparsity.nnz, self.sparsity.onnz))
        mat.setLGMap(rmap=row_lg, cmap=col_lg)
        # Do not stash entries destined for other processors, just drop them
        # (we take care of those in the halo)
        mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, True)
        # Any add or insertion that would generate a new entry that has not
        # been preallocated will raise an error
        mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        # When zeroing rows (e.g. for enforcing Dirichlet bcs), keep those in
        # the nonzero structure of the matrix. Otherwise PETSc would compact
        # the sparsity and render our sparsity caching useless.
        mat.setOption(mat.Option.KEEP_NONZERO_PATTERN, True)
        self._handle = mat
        # Matrices start zeroed.
        self._version_set_zero()

    def __getitem__(self, idx):
        """Return :class:`Mat` block with row and column given by ``idx``
        or a given row of blocks."""
        try:
            i, j = idx
            return self.blocks[i][j]
        except TypeError:
            return self.blocks[idx]

    def __iter__(self):
        """Iterate over all :class:`Mat` blocks by row and then by column."""
        for row in self.blocks:
            for s in row:
                yield s

    @collective
    def dump(self, filename):
        """Dump the matrix to file ``filename`` in PETSc binary format."""
        base._trace.evaluate(set([self]), set())
        vwr = PETSc.Viewer().createBinary(filename, PETSc.Viewer.Mode.WRITE)
        self.handle.view(vwr)

    @collective
    def zero(self):
        """Zero the matrix."""
        base._trace.evaluate(set(), set([self]))
        self.handle.zeroEntries()
        self._version_set_zero()

    @modifies
    @collective
    def zero_rows(self, rows, diag_val=1.0):
        """Zeroes the specified rows of the matrix, with the exception of the
        diagonal entry, which is set to diag_val. May be used for applying
        strong boundary conditions.

        :param rows: a :class:`Subset` or an iterable"""
        base._trace.evaluate(set([self]), set([self]))
        rows = rows.indices if isinstance(rows, Subset) else rows
        self.handle.zeroRowsLocal(rows, diag_val)

    @collective
    def set_diagonal(self, vec):
        """Add a vector to the diagonal of the matrix.

        :params vec: vector to add (:class:`Dat` or :class:`PETsc.Vec`)"""
        if self.sparsity.shape != (1, 1):
            if not isinstance(vec, base.MixedDat):
                raise TypeError('Can only set diagonal of blocked Mat from MixedDat')
            if vec.dataset != self.sparsity.dsets[1]:
                raise TypeError('Mismatching datasets for MixedDat and Mat')
            rows, cols = self.sparsity.shape
            for i in range(rows):
                if i < cols:
                    self[i, i].set_diagonal(vec[i])
            return
        r, c = self.handle.getSize()
        if r != c:
            raise MatTypeError('Cannot set diagonal of non-square matrix')
        if not isinstance(vec, (base.Dat, PETSc.Vec)):
            raise TypeError("Can only set diagonal from a Dat or PETSc Vec.")
        if isinstance(vec, PETSc.Vec):
            self.handle.setDiagonal(vec)
        else:
            with vec.vec_ro as v:
                self.handle.setDiagonal(v)

    def _cow_actual_copy(self, src):
        self._handle = src.handle.duplicate(copy=True)
        return self

    @collective
    def inc_local_diagonal_entries(self, rows, diag_val=1.0):
        """Increment the diagonal entry in ``rows`` by a particular value.

        :param rows: a :class:`Subset` or an iterable.
        :param diag_val: the value to add

        The indices in ``rows`` should index the process-local rows of
        the matrix (no mapping to global indexes is applied).

        The diagonal entries corresponding to the complement of rows
        are incremented by zero.
        """
        base._trace.evaluate(set([self]), set([self]))
        vec = self.handle.createVecLeft()
        vec.setOption(vec.Option.IGNORE_OFF_PROC_ENTRIES, True)
        with vec as array:
            rows = rows[rows < self.sparsity.rmaps[0].toset.size]
            array[rows] = diag_val
        self.handle.setDiagonal(vec, addv=PETSc.InsertMode.ADD_VALUES)

    @collective
    def _assemble(self):
        if not self._ever_assembled and MPI.parallel:
            # add zero to diagonal entries (so they're not compressed out
            # in the assembly).  This is necessary for parallel where we
            # currently don't give an exact sparsity pattern.
            rows, cols = self.sparsity.shape
            for i in range(rows):
                if i < cols:
                    v = self[i, i].handle.createVecLeft()
                    self[i, i].handle.setDiagonal(v, addv=PETSc.InsertMode.ADD_VALUES)
            self._ever_assembled = True
        # Now that we've filled up the sparsity pattern, we can ignore
        # zero entries for MatSetValues calls.
        # Do not create a zero location when adding a zero value
        self._handle.setOption(self._handle.Option.IGNORE_ZERO_ENTRIES, True)
        self.handle.assemble()

    @property
    def blocks(self):
        """2-dimensional array of matrix blocks."""
        if not hasattr(self, '_blocks'):
            self._init()
        return self._blocks

    @property
    def array(self):
        """Array of non-zero values."""
        if not hasattr(self, '_array'):
            self._init()
        base._trace.evaluate(set([self]), set())
        return self._array

    @property
    def values(self):
        base._trace.evaluate(set([self]), set())
        return self.handle[:, :]

    @property
    def handle(self):
        """Petsc4py Mat holding matrix data."""
        if not hasattr(self, '_handle'):
            self._init()
        return self._handle

    def __mul__(self, v):
        """Multiply this :class:`Mat` with the vector ``v``."""
        if not isinstance(v, (base.Dat, PETSc.Vec)):
            raise TypeError("Can only multiply Mat and Dat or PETSc Vec.")
        if isinstance(v, base.Dat):
            with v.vec_ro as vec:
                y = self.handle * vec
        else:
            y = self.handle * v
        if isinstance(v, base.MixedDat):
            dat = _make_object('MixedDat', self.sparsity.dsets[0])
            offset = 0
            for d in dat:
                sz = d.dataset.set.size
                d.data[:] = y.getSubVector(PETSc.IS().createStride(sz, offset, 1)).array[:]
                offset += sz
        else:
            dat = _make_object('Dat', self.sparsity.dsets[0])
            dat.data[:] = y.array[:]
        dat.needs_halo_update = True
        return dat

# FIXME: Eventually (when we have a proper OpenCL solver) this wants to go in
# sequential


class Solver(base.Solver, PETSc.KSP):

    _cnt = 0

    def __init__(self, parameters=None, **kwargs):
        super(Solver, self).__init__(parameters, **kwargs)
        self._count = Solver._cnt
        Solver._cnt += 1
        self.create(PETSc.COMM_WORLD)
        prefix = 'pyop2_ksp_%d' % self._count
        self.setOptionsPrefix(prefix)
        converged_reason = self.ConvergedReason()
        self._reasons = dict([(getattr(converged_reason, r), r)
                              for r in dir(converged_reason)
                              if not r.startswith('_')])

    @collective
    def _set_parameters(self):
        opts = PETSc.Options()
        opts.prefix = self.getOptionsPrefix()
        for k, v in self.parameters.iteritems():
            if type(v) is bool:
                if v:
                    opts[k] = None
                else:
                    continue
            else:
                opts[k] = v
        self.setFromOptions()
        for k in self.parameters.iterkeys():
            del opts[k]

    @collective
    def _solve(self, A, x, b):
        self.setOperators(A.handle)
        self._set_parameters()
        if self.parameters['plot_convergence']:
            self.reshist = []

            def monitor(ksp, its, norm):
                self.reshist.append(norm)
                debug("%3d KSP Residual norm %14.12e" % (its, norm))
            self.setMonitor(monitor)
        # Not using super here since the MRO would call base.Solver.solve
        with b.vec_ro as bv:
            with x.vec as xv:
                PETSc.KSP.solve(self, bv, xv)
        if self.parameters['plot_convergence']:
            self.cancelMonitor()
            try:
                import pylab
                pylab.semilogy(self.reshist)
                pylab.title('Convergence history')
                pylab.xlabel('Iteration')
                pylab.ylabel('Residual norm')
                pylab.savefig('%sreshist_%04d.png' %
                              (self.parameters['plot_prefix'], self._count))
            except ImportError:
                warning("pylab not available, not plotting convergence history.")
        r = self.getConvergedReason()
        debug("Converged reason: %s" % self._reasons[r])
        debug("Iterations: %s" % self.getIterationNumber())
        debug("Residual norm: %s" % self.getResidualNorm())
        if r < 0:
            msg = "KSP Solver failed to converge in %d iterations: %s (Residual norm: %e)" \
                % (self.getIterationNumber(), self._reasons[r], self.getResidualNorm())
            if self.parameters['error_on_nonconvergence']:
                raise RuntimeError(msg)
            else:
                warning(msg)
