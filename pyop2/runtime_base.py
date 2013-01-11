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

""" Base classes for OP2 objects. The versions here extend those from the :mod:`base` module to include runtime data information which is backend independent. Individual runtime backends should subclass these as required to implement backend-specific features."""

import numpy as np

from exceptions import *
from utils import *
import configuration as cfg
import base
from base import READ, WRITE, RW, INC, MIN, MAX, IterationSpace
from base import DataCarrier, IterationIndex, i, IdentityMap, Kernel, Global
from base import _parloop_cache, _empty_parloop_cache, _parloop_cache_size
import op_lib_core as core
from petsc4py import PETSc

# Data API

class Arg(base.Arg):
    """An argument to a :func:`par_loop`.

    .. warning:: User code should not directly instantiate :class:`Arg`. Instead, use the call syntax on the :class:`DataCarrier`.
    """

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_arg(self)
        return self._lib_handle

class Set(base.Set):
    """OP2 set."""

    @validate_type(('size', int, SizeTypeError))
    def __init__(self, size, name=None):
        base.Set.__init__(self, size, name)

    @classmethod
    def fromhdf5(cls, f, name):
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

class Dat(base.Dat):
    """OP2 vector data. A ``Dat`` holds a value for every member of a :class:`Set`."""

    @classmethod
    def fromhdf5(cls, dataset, f, name):
        slot = f[name]
        data = slot.value
        dim = slot.shape[1:]
        soa = slot.attrs['type'].find(':soa') > 0
        if len(dim) < 1:
            raise DimTypeError("Invalid dimension value %s" % dim)
        ret = cls(dataset, dim, data, name=name, soa=soa)
        return ret

    @property
    def _c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_dat(self)
        return self._lib_handle

class Const(base.Const):
    """Data that is constant for any element of any set."""

    @classmethod
    def fromhdf5(cls, f, name):
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
        super(Sparsity, self).__init__(maps, dims, name)
        key = (maps, as_tuple(dims, int, 2))
        self._cached = True
        core.build_sparsity(self)
        self._total_nz = self._rowptr[-1]
        _sparsity_cache[key] = self

    def __del__(self):
        core.free_sparsity(self)

    @property
    def rowptr(self):
        return self._rowptr

    @property
    def colidx(self):
        return self._colidx

    @property
    def d_nnz(self):
        return self._d_nnz

    @property
    def total_nz(self):
        return int(self._total_nz)

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
        mat = PETSc.Mat()
        rdim, cdim = self.sparsity.dims
        self._array = np.zeros(self.sparsity.total_nz, dtype=PETSc.RealType)
        # We're not currently building a blocked matrix, so need to scale the
        # number of rows and columns by the sparsity dimensions
        # FIXME: This needs to change if we want to do blocked sparse
        mat.createAIJWithArrays((self.sparsity.nrows*rdim, self.sparsity.ncols*cdim),
                (self.sparsity._rowptr, self.sparsity._colidx, self._array))
        self._handle = mat

    def zero(self):
        """Zero the matrix."""
        self.handle.zeroEntries()

    def zero_rows(self, rows, diag_val):
        """Zeroes the specified rows of the matrix, with the exception of the
        diagonal entry, which is set to diag_val. May be used for applying
        strong boundary conditions."""
        self.handle.zeroRows(rows, diag_val)

    def _assemble(self):
        self.handle.assemble()

    @property
    def array(self):
        if not hasattr(self, '_array'):
            self._init()
        return self._array

    @property
    def values(self):
        return self.handle[:,:]

    @property
    def handle(self):
        if self._handle is None:
            self._init()
        return self._handle

class ParLoop(base.ParLoop):
    def compute(self):
        raise RuntimeError('Must select a backend')

# FIXME: Eventually (when we have a proper OpenCL solver) this wants to go in
# sequential
class Solver(base.Solver, PETSc.KSP):

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

    def solve(self, A, x, b):
        self._set_parameters()
        px = PETSc.Vec().createWithArray(x.data)
        pb = PETSc.Vec().createWithArray(b.data)
        self.setOperators(A.handle)
        # Not using super here since the MRO would call base.Solver.solve
        PETSc.KSP.solve(self, pb, px)
        r = self.getConvergedReason()
        if cfg.debug:
            print "Converged reason: %s" % self._reasons[r]
            print "Iterations: %s" % self.getIterationNumber()
        if r < 0:
            if self.parameters['error_on_nonconvergence']:
                raise RuntimeError("KSP Solver failed to converge: %s" % self._reasons[r])
            else:
                from warnings import warn
                warn("KSP Solver failed to converge: %s" % self._reasons[r])
