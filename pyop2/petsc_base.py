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
from base import _parloop_cache, _empty_parloop_cache, _parloop_cache_size

def set_mpi_communicator(comm):
    base.set_mpi_communicator(comm)
    # PETSc objects also need to be built on the same communicator.
    PETSc.Sys.setDefaultComm(base.PYOP2_COMM)

class Dat(base.Dat):

    @property
    def vec(self):
        """PETSc Vec appropriate for this Dat."""
        if not hasattr(self, '_vec'):
            size = (self.dataset.size * self.cdim, None)
            self._vec = PETSc.Vec().createWithArray(self._data, size=size)
        return self._vec


class Mat(base.Mat):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    def _init(self):
        if not self.dtype == PETSc.ScalarType:
            raise RuntimeError("Can only create a matrix of type %s, %s is not supported" \
                    % (PETSc.ScalarType, self.dtype))
        if base.PYOP2_COMM.size == 1:
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
