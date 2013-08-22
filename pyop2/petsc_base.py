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

from petsc4py import PETSc
import base
from base import *
from logger import debug
import mpi
from mpi import collective


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

    @property
    @collective
    def vec(self):
        """PETSc Vec appropriate for this Dat."""
        if not hasattr(self, '_vec'):
            size = (self.dataset.size * self.cdim, None)
            self._vec = PETSc.Vec().createWithArray(self._data, size=size)
        return self._vec


class Mat(base.Mat):

    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    @collective
    def _init(self):
        if not self.dtype == PETSc.ScalarType:
            raise RuntimeError("Can only create a matrix of type %s, %s is not supported"
                               % (PETSc.ScalarType, self.dtype))
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
            mat.createAIJ(size=((self.sparsity.nrows * rdim, None),
                                (self.sparsity.ncols * cdim, None)),
                          nnz=(self.sparsity.nnz, self.sparsity.onnz))
        mat.setLGMap(rmap=row_lg, cmap=col_lg)
        # Do not stash entries destined for other processors, just drop them
        # (we take care of those in the halo)
        mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, True)
        # Do not create a zero location when adding a zero value
        mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)
        # Any add or insertion that would generate a new entry that has not
        # been preallocated will raise an error
        mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        # When zeroing rows (e.g. for enforcing Dirichlet bcs), keep those in
        # the nonzero structure of the matrix. Otherwise PETSc would compact
        # the sparsity and render our sparsity caching useless.
        mat.setOption(mat.Option.KEEP_NONZERO_PATTERN, True)
        self._handle = mat

    @collective
    def dump(self, filename):
        """Dump the matrix to file ``filename`` in PETSc binary format."""
        vwr = PETSc.Viewer().createBinary(filename, PETSc.Viewer.Mode.WRITE)
        self.handle.view(vwr)

    @collective
    def zero(self):
        """Zero the matrix."""
        self.handle.zeroEntries()

    @collective
    def zero_rows(self, rows, diag_val):
        """Zeroes the specified rows of the matrix, with the exception of the
        diagonal entry, which is set to diag_val. May be used for applying
        strong boundary conditions."""
        self.handle.zeroRowsLocal(rows, diag_val)

    @collective
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
        return self.handle[:, :]

    @property
    def handle(self):
        """Petsc4py Mat holding matrix data."""
        if not hasattr(self, '_handle'):
            self._init()
        return self._handle

# FIXME: Eventually (when we have a proper OpenCL solver) this wants to go in
# sequential


class Solver(base.Solver, PETSc.KSP):

    _cnt = 0

    def __init__(self, parameters=None, **kwargs):
        super(Solver, self).__init__(parameters, **kwargs)
        self.create(PETSc.COMM_WORLD)
        converged_reason = self.ConvergedReason()
        self._reasons = dict([(getattr(converged_reason, r), r)
                              for r in dir(converged_reason)
                              if not r.startswith('_')])

    @collective
    def _set_parameters(self):
        self.setType(self.parameters['linear_solver'])
        self.getPC().setType(self.parameters['preconditioner'])
        self.rtol = self.parameters['relative_tolerance']
        self.atol = self.parameters['absolute_tolerance']
        self.divtol = self.parameters['divergence_tolerance']
        self.max_it = self.parameters['maximum_iterations']
        if self.parameters['plot_convergence']:
            self.parameters['monitor_convergence'] = True

    @collective
    def solve(self, A, x, b):
        self._set_parameters()
        self.setOperators(A.handle)
        self.setFromOptions()
        if self.parameters['monitor_convergence']:
            self.reshist = []

            def monitor(ksp, its, norm):
                self.reshist.append(norm)
                debug("%3d KSP Residual norm %14.12e" % (its, norm))
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
                    pylab.savefig('%sreshist_%04d.png' %
                                  (self.parameters['plot_prefix'], Solver._cnt))
                    Solver._cnt += 1
                except ImportError:
                    from warnings import warn
                    warn("pylab not available, not plotting convergence history.")
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
                from warnings import warn
                warn(msg)
