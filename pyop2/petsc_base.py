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

from contextlib import contextmanager
from petsc4py import PETSc
import itertools
import numpy as np
import abc

from pyop2.datatypes import IntType, ScalarType
from pyop2 import base
from pyop2 import mpi
from pyop2 import sparsity
from pyop2 import utils
from pyop2.base import _make_object, Subset
from pyop2.mpi import collective
from pyop2.profiling import timed_region


class DataSet(base.DataSet):

    @utils.cached_property
    def lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`DataSet`.
        """
        lgmap = PETSc.LGMap()
        if self.comm.size == 1:
            lgmap.create(indices=np.arange(self.size, dtype=IntType),
                         bsize=self.cdim, comm=self.comm)
        else:
            lgmap.create(indices=self.halo.local_to_global_numbering,
                         bsize=self.cdim, comm=self.comm)
        return lgmap

    @utils.cached_property
    def scalar_lgmap(self):
        if self.cdim == 1:
            return self.lgmap
        indices = self.lgmap.block_indices
        return PETSc.LGMap().create(indices=indices, bsize=1, comm=self.comm)

    @utils.cached_property
    def unblocked_lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`DataSet` with a block size of 1.
        """
        if self.cdim == 1:
            return self.lgmap
        else:
            indices = self.lgmap.indices
            lgmap = PETSc.LGMap().create(indices=indices,
                                         bsize=1, comm=self.lgmap.comm)
            return lgmap

    @utils.cached_property
    def field_ises(self):
        """A list of PETSc ISes defining the global indices for each set in
        the DataSet.

        Used when extracting blocks from matrices for solvers."""
        ises = []
        nlocal_rows = 0
        for dset in self:
            nlocal_rows += dset.size * dset.cdim
        offset = self.comm.scan(nlocal_rows)
        offset -= nlocal_rows
        for dset in self:
            nrows = dset.size * dset.cdim
            iset = PETSc.IS().createStride(nrows, first=offset, step=1,
                                           comm=self.comm)
            iset.setBlockSize(dset.cdim)
            ises.append(iset)
            offset += nrows
        return tuple(ises)

    @utils.cached_property
    def local_ises(self):
        """A list of PETSc ISes defining the local indices for each set in the DataSet.

        Used when extracting blocks from matrices for assembly."""
        ises = []
        start = 0
        for dset in self:
            bs = dset.cdim
            n = dset.total_size*bs
            iset = PETSc.IS().createStride(n, first=start, step=1,
                                           comm=mpi.COMM_SELF)
            iset.setBlockSize(bs)
            start += n
            ises.append(iset)
        return tuple(ises)

    @utils.cached_property
    def layout_vec(self):
        """A PETSc Vec compatible with the dof layout of this DataSet."""
        vec = PETSc.Vec().create(comm=self.comm)
        size = (self.size * self.cdim, None)
        vec.setSizes(size, bsize=self.cdim)
        vec.setUp()
        return vec

    @utils.cached_property
    def dm(self):
        dm = PETSc.DMShell().create(comm=self.comm)
        dm.setGlobalVector(self.layout_vec)
        return dm


class GlobalDataSet(base.GlobalDataSet):

    @utils.cached_property
    def lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`DataSet`.
        """
        lgmap = PETSc.LGMap()
        lgmap.create(indices=np.arange(1, dtype=IntType),
                     bsize=self.cdim, comm=self.comm)
        return lgmap

    @utils.cached_property
    def unblocked_lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`DataSet` with a block size of 1.
        """
        if self.cdim == 1:
            return self.lgmap
        else:
            indices = self.lgmap.indices
            lgmap = PETSc.LGMap().create(indices=indices,
                                         bsize=1, comm=self.lgmap.comm)
            return lgmap

    @utils.cached_property
    def field_ises(self):
        """A list of PETSc ISes defining the global indices for each set in
        the DataSet.

        Used when extracting blocks from matrices for solvers."""
        ises = []
        nlocal_rows = 0
        for dset in self:
            nlocal_rows += dset.size * dset.cdim
        offset = self.comm.scan(nlocal_rows)
        offset -= nlocal_rows
        for dset in self:
            nrows = dset.size * dset.cdim
            iset = PETSc.IS().createStride(nrows, first=offset, step=1,
                                           comm=self.comm)
            iset.setBlockSize(dset.cdim)
            ises.append(iset)
            offset += nrows
        return tuple(ises)

    @utils.cached_property
    def local_ises(self):
        """A list of PETSc ISes defining the local indices for each set in the DataSet.

        Used when extracting blocks from matrices for assembly."""
        raise NotImplementedError

    @utils.cached_property
    def layout_vec(self):
        """A PETSc Vec compatible with the dof layout of this DataSet."""
        vec = PETSc.Vec().create(comm=self.comm)
        size = (self.size * self.cdim, None)
        vec.setSizes(size, bsize=self.cdim)
        vec.setUp()
        return vec

    @utils.cached_property
    def dm(self):
        dm = PETSc.DMShell().create(comm=self.comm)
        dm.setGlobalVector(self.layout_vec)
        return dm


class MixedDataSet(DataSet, base.MixedDataSet):

    @utils.cached_property
    def layout_vec(self):
        """A PETSc Vec compatible with the dof layout of this MixedDataSet."""
        vec = PETSc.Vec().create(comm=self.comm)
        # Compute local and global size from sizes of layout vecs
        lsize, gsize = map(sum, zip(*(d.layout_vec.sizes for d in self)))
        vec.setSizes((lsize, gsize), bsize=1)
        vec.setUp()
        return vec

    @utils.cached_property
    def lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`MixedDataSet`.
        """
        lgmap = PETSc.LGMap()
        if self.comm.size == 1:
            size = sum(s.size * s.cdim for s in self)
            lgmap.create(indices=np.arange(size, dtype=IntType),
                         bsize=1, comm=self.comm)
            return lgmap
        # Compute local to global maps for a monolithic mixed system
        # from the individual local to global maps for each field.
        # Exposition:
        #
        # We have N fields and P processes.  The global row
        # ordering is:
        #
        # f_0_p_0, f_1_p_0, ..., f_N_p_0; f_0_p_1, ..., ; f_0_p_P,
        # ..., f_N_p_P.
        #
        # We have per-field local to global numberings, to convert
        # these into multi-field local to global numberings, we note
        # the following:
        #
        # For each entry in the per-field l2g map, we first determine
        # the rank that entry belongs to, call this r.
        #
        # We know that this must be offset by:
        # 1. The sum of all field lengths with rank < r
        # 2. The sum of all lower-numbered field lengths on rank r.
        #
        # Finally, we need to shift the field-local entry by the
        # current field offset.
        idx_size = sum(s.total_size*s.cdim for s in self)
        indices = np.full(idx_size, -1, dtype=IntType)
        owned_sz = np.array([sum(s.size * s.cdim for s in self)],
                            dtype=IntType)
        field_offset = np.empty_like(owned_sz)
        self.comm.Scan(owned_sz, field_offset)
        field_offset -= owned_sz

        all_field_offsets = np.empty(self.comm.size, dtype=IntType)
        self.comm.Allgather(field_offset, all_field_offsets)

        start = 0
        all_local_offsets = np.zeros(self.comm.size, dtype=IntType)
        current_offsets = np.zeros(self.comm.size + 1, dtype=IntType)
        for s in self:
            idx = indices[start:start + s.total_size * s.cdim]
            owned_sz[0] = s.size * s.cdim
            self.comm.Scan(owned_sz, field_offset)
            self.comm.Allgather(field_offset, current_offsets[1:])
            # Find the ranks each entry in the l2g belongs to
            l2g = s.halo.local_to_global_numbering
            # If cdim > 1, we need to unroll the node numbering to dof
            # numbering
            if s.cdim > 1:
                new_l2g = np.empty(l2g.shape[0]*s.cdim, dtype=l2g.dtype)
                for i in range(s.cdim):
                    new_l2g[i::s.cdim] = l2g*s.cdim + i
                l2g = new_l2g
            tmp_indices = np.searchsorted(current_offsets, l2g, side="right") - 1
            idx[:] = l2g[:] - current_offsets[tmp_indices] + \
                all_field_offsets[tmp_indices] + all_local_offsets[tmp_indices]
            self.comm.Allgather(owned_sz, current_offsets[1:])
            all_local_offsets += current_offsets[1:]
            start += s.total_size * s.cdim
        lgmap.create(indices=indices, bsize=1, comm=self.comm)
        return lgmap

    @utils.cached_property
    def unblocked_lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`DataSet` with a block size of 1.
        """
        return self.lgmap


class VecAccessMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def vec_context(self, access):
        pass

    @abc.abstractproperty
    def _vec(self):
        pass

    @property
    @collective
    def vec(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're allowed to modify the data you get back from this view."""
        return self.vec_context(access=base.RW)

    @property
    @collective
    def vec_wo(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're allowed to modify the data you get back from this view,
        but you cannot read from it."""
        return self.vec_context(access=base.WRITE)

    @property
    @collective
    def vec_ro(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're not allowed to modify the data you get back from this view."""
        return self.vec_context(access=base.READ)


class Dat(base.Dat, VecAccessMixin):
    @utils.cached_property
    def _vec(self):
        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # Can't duplicate layout_vec of dataset, because we then
        # carry around extra unnecessary data.
        # But use getSizes to save an Allreduce in computing the
        # global size.
        size = self.dataset.layout_vec.getSizes()
        data = self._data[:size[0]]
        return PETSc.Vec().createWithArray(data, size=size, bsize=self.cdim, comm=self.comm)

    @contextmanager
    def vec_context(self, access):
        r"""A context manager for a :class:`PETSc.Vec` from a :class:`Dat`.

        :param access: Access descriptor: READ, WRITE, or RW."""
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec
        if access is not base.READ:
            self.halo_valid = False


class MixedDat(base.MixedDat, VecAccessMixin):
    @utils.cached_property
    def _vec(self):
        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # In this case we can just duplicate the layout vec
        # because we're not placing an array.
        return self.dataset.layout_vec.duplicate()

    @contextmanager
    def vec_context(self, access):
        r"""A context manager scattering the arrays of all components of this
        :class:`MixedDat` into a contiguous :class:`PETSc.Vec` and reverse
        scattering to the original arrays when exiting the context.

        :param access: Access descriptor: READ, WRITE, or RW.

        .. note::

           The :class:`~PETSc.Vec` obtained from this context is in
           the correct order to be left multiplied by a compatible
           :class:`MixedMat`.  In parallel it is *not* just a
           concatenation of the underlying :class:`Dat`\s."""
        # Do the actual forward scatter to fill the full vector with
        # values
        if access is not base.WRITE:
            offset = 0
            array = self._vec.array
            for d in self:
                with d.vec_ro as v:
                    size = v.local_size
                    array[offset:offset+size] = v.array_r[:]
                    offset += size
        self._vec.stateIncrease()
        yield self._vec
        if access is not base.READ:
            # Reverse scatter to get the values back to their original locations
            offset = 0
            array = self._vec.array_r
            for d in self:
                with d.vec_wo as v:
                    size = v.local_size
                    v.array[:] = array[offset:offset+size]
                    offset += size
            self.halo_valid = False


class Global(base.Global, VecAccessMixin):
    @utils.cached_property
    def _vec(self):
        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        # Can't duplicate layout_vec of dataset, because we then
        # carry around extra unnecessary data.
        # But use getSizes to save an Allreduce in computing the
        # global size.
        data = self._data
        size = self.dataset.layout_vec.getSizes()
        if self.comm.rank == 0:
            return PETSc.Vec().createWithArray(data, size=size,
                                               bsize=self.cdim,
                                               comm=self.comm)
        else:
            return PETSc.Vec().createWithArray(np.empty(0, dtype=self.dtype),
                                               size=size,
                                               bsize=self.cdim,
                                               comm=self.comm)

    @contextmanager
    def vec_context(self, access):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Global`.

        :param access: Access descriptor: READ, WRITE, or RW."""
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec
        if access is not base.READ:
            data = self._data
            self.comm.Bcast(data, 0)


class SparsityBlock(base.Sparsity):
    """A proxy class for a block in a monolithic :class:`.Sparsity`.

    :arg parent: The parent monolithic sparsity.
    :arg i: The block row.
    :arg j: The block column.

    .. warning::

       This class only implements the properties necessary to infer
       its shape.  It does not provide arrays of non zero fill."""
    def __init__(self, parent, i, j):
        self._dsets = (parent.dsets[0][i], parent.dsets[1][j])
        self._rmaps = tuple(m.split[i] for m in parent.rmaps)
        self._cmaps = tuple(m.split[j] for m in parent.cmaps)
        self._nrows = self._dsets[0].size
        self._ncols = self._dsets[1].size
        self._has_diagonal = i == j and parent._has_diagonal
        self._parent = parent
        self._dims = tuple([tuple([parent.dims[i][j]])])
        self._blocks = [[self]]
        self.iteration_regions = parent.iteration_regions
        self.lcomm = self.dsets[0].comm
        self.rcomm = self.dsets[1].comm
        # TODO: think about lcomm != rcomm
        self.comm = self.lcomm

    @classmethod
    def _process_args(cls, *args, **kwargs):
        return (None, ) + args, kwargs

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        return None

    def __repr__(self):
        return "SparsityBlock(%r, %r, %r)" % (self._parent, self._i, self._j)


def masked_lgmap(lgmap, mask, block=True):
    if block:
        indices = lgmap.block_indices.copy()
        bsize = lgmap.getBlockSize()
    else:
        indices = lgmap.indices.copy()
        bsize = 1
    indices[mask] = -1
    return PETSc.LGMap().create(indices=indices, bsize=bsize, comm=lgmap.comm)


class MatBlock(base.Mat):
    """A proxy class for a local block in a monolithic :class:`.Mat`.

    :arg parent: The parent monolithic matrix.
    :arg i: The block row.
    :arg j: The block column.
    """
    def __init__(self, parent, i, j):
        self._parent = parent
        self._i = i
        self._j = j
        self._sparsity = SparsityBlock(parent.sparsity, i, j)
        rset, cset = self._parent.sparsity.dsets
        rowis = rset.local_ises[i]
        colis = cset.local_ises[j]
        self.handle = parent.handle.getLocalSubMatrix(isrow=rowis,
                                                      iscol=colis)
        self.comm = parent.comm
        self.local_to_global_maps = self.handle.getLGMap()

    @utils.cached_property
    def _kernel_args_(self):
        return (self.handle.handle, )

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (type(self._parent), self._parent.dtype, self.dims)

    @property
    def assembly_state(self):
        # Track our assembly state only
        return self._parent.assembly_state

    @assembly_state.setter
    def assembly_state(self, value):
        self._parent.assembly_state = value

    def __getitem__(self, idx):
        return self

    def __iter__(self):
        yield self

    def _flush_assembly(self):
        # Need to flush for all blocks
        for b in self._parent:
            b.handle.assemble(assembly=PETSc.Mat.AssemblyType.FLUSH)
        self._parent._flush_assembly()

    def set_local_diagonal_entries(self, rows, diag_val=1.0, idx=None):
        rows = np.asarray(rows, dtype=IntType)
        rbs, _ = self.dims[0][0]
        if rbs > 1:
            if idx is not None:
                rows = rbs * rows + idx
            else:
                rows = np.dstack([rbs*rows + i for i in range(rbs)]).flatten()
        rows = rows.reshape(-1, 1)
        self.change_assembly_state(Mat.INSERT_VALUES)
        if len(rows) > 0:
            values = np.full(rows.shape, diag_val, dtype=ScalarType)
            self.handle.setValuesLocalRCV(rows, rows, values,
                                          addv=PETSc.InsertMode.INSERT_VALUES)

    def addto_values(self, rows, cols, values):
        """Add a block of values to the :class:`Mat`."""
        self.change_assembly_state(Mat.ADD_VALUES)
        if len(values) > 0:
            self.handle.setValuesBlockedLocal(rows, cols, values,
                                              addv=PETSc.InsertMode.ADD_VALUES)

    def set_values(self, rows, cols, values):
        """Set a block of values in the :class:`Mat`."""
        self.change_assembly_state(Mat.INSERT_VALUES)
        if len(values) > 0:
            self.handle.setValuesBlockedLocal(rows, cols, values,
                                              addv=PETSc.InsertMode.INSERT_VALUES)

    def assemble(self):
        raise RuntimeError("Should never call assemble on MatBlock")

    @property
    def values(self):
        rset, cset = self._parent.sparsity.dsets
        rowis = rset.field_ises[self._i]
        colis = cset.field_ises[self._j]
        self._parent.assemble()
        mat = self._parent.handle.createSubMatrix(isrow=rowis,
                                                  iscol=colis)
        return mat[:, :]

    @property
    def dtype(self):
        return self._parent.dtype

    @property
    def nbytes(self):
        return self._parent.nbytes // (np.prod(self.sparsity.shape))

    def __repr__(self):
        return "MatBlock(%r, %r, %r)" % (self._parent, self._i, self._j)

    def __str__(self):
        return "Block[%s, %s] of %s" % (self._i, self._j, self._parent)


class Mat(base.Mat):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    def __init__(self, *args, **kwargs):
        self.mat_type = kwargs.pop("mat_type", None)
        base.Mat.__init__(self, *args, **kwargs)
        self._init()
        self.assembly_state = Mat.ASSEMBLED

    # Firedrake relies on this to distinguish between MatBlock and not for boundary conditions
    local_to_global_maps = (None, None)

    @utils.cached_property
    def _kernel_args_(self):
        return tuple(a.handle.handle for a in self)

    @collective
    def _init(self):
        if not self.dtype == PETSc.ScalarType:
            raise RuntimeError("Can only create a matrix of type %s, %s is not supported"
                               % (PETSc.ScalarType, self.dtype))
        if self.mat_type == "dense":
            self._init_dense()
        # If the Sparsity is defined on MixedDataSets, we need to build a MatNest
        elif self.sparsity.shape > (1, 1):
            if self.sparsity.nested:
                self._init_nest()
                self._nested = True
            else:
                self._init_monolithic()
        else:
            self._init_block()

    def _init_dense(self):
        mat = PETSc.Mat()
        rset, cset = self.sparsity.dsets
        rlgmap = rset.unblocked_lgmap
        clgmap = cset.unblocked_lgmap
        mat.createDense(size=((self.nrows, None), (self.ncols, None)),
                        bsize=1,
                        comm=self.comm)
        mat.setLGMap(rmap=rlgmap, cmap=clgmap)
        self.handle = mat
        self._blocks = []
        rows, cols = self.sparsity.shape
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(MatBlock(self, i, j))
            self._blocks.append(row)
        mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, False)
        mat.setOption(mat.Option.SUBSET_OFF_PROC_ENTRIES, True)
        mat.setUp()
        # Put zeros in all the places we might eventually put a value.
        with timed_region("MatZeroInitial"):
            mat.zeroEntries()
        mat.assemble()

    def _init_monolithic(self):
        mat = PETSc.Mat()
        rset, cset = self.sparsity.dsets
        rlgmap = rset.unblocked_lgmap
        clgmap = cset.unblocked_lgmap
        mat.createAIJ(size=((self.nrows, None), (self.ncols, None)),
                      nnz=(self.sparsity.nnz, self.sparsity.onnz),
                      bsize=1,
                      comm=self.comm)
        mat.setLGMap(rmap=rlgmap, cmap=clgmap)
        self.handle = mat
        self._blocks = []
        rows, cols = self.sparsity.shape
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(MatBlock(self, i, j))
            self._blocks.append(row)
        mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, False)
        mat.setOption(mat.Option.KEEP_NONZERO_PATTERN, True)
        # We completely fill the allocated matrix when zeroing the
        # entries, so raise an error if we "missed" one.
        mat.setOption(mat.Option.UNUSED_NONZERO_LOCATION_ERR, True)
        mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, False)
        mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        # The first assembly (filling with zeros) sets all possible entries.
        mat.setOption(mat.Option.SUBSET_OFF_PROC_ENTRIES, True)
        # Put zeros in all the places we might eventually put a value.
        with timed_region("MatZeroInitial"):
            for i in range(rows):
                for j in range(cols):
                    sparsity.fill_with_zeros(self[i, j].handle,
                                             self[i, j].sparsity.dims[0][0],
                                             self[i, j].sparsity.maps,
                                             self[i, j].sparsity.iteration_regions,
                                             set_diag=self[i, j].sparsity._has_diagonal)
                    self[i, j].handle.assemble()

        mat.assemble()
        mat.setOption(mat.Option.NEW_NONZERO_LOCATION_ERR, True)
        mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)

    def _init_nest(self):
        mat = PETSc.Mat()
        self._blocks = []
        rows, cols = self.sparsity.shape
        rset, cset = self.sparsity.dsets
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(Mat(self.sparsity[i, j], self.dtype,
                           '_'.join([self.name, str(i), str(j)])))
            self._blocks.append(row)
        # PETSc Mat.createNest wants a flattened list of Mats
        mat.createNest([[m.handle for m in row_] for row_ in self._blocks],
                       isrows=rset.field_ises, iscols=cset.field_ises,
                       comm=self.comm)
        self.handle = mat

    def _init_block(self):
        self._blocks = [[self]]

        rset, cset = self.sparsity.dsets
        if (isinstance(rset, GlobalDataSet) or isinstance(cset, GlobalDataSet)):
            self._init_global_block()
            return

        mat = PETSc.Mat()
        row_lg = rset.lgmap
        col_lg = cset.lgmap
        rdim, cdim = self.dims[0][0]

        if rdim == cdim and rdim > 1 and self.sparsity._block_sparse:
            # Size is total number of rows and columns, but the
            # /sparsity/ is the block sparsity.
            block_sparse = True
            create = mat.createBAIJ
        else:
            # Size is total number of rows and columns, sparsity is
            # the /dof/ sparsity.
            block_sparse = False
            create = mat.createAIJ
        create(size=((self.nrows, None),
                     (self.ncols, None)),
               nnz=(self.sparsity.nnz, self.sparsity.onnz),
               bsize=(rdim, cdim),
               comm=self.comm)
        mat.setLGMap(rmap=row_lg, cmap=col_lg)
        # Stash entries destined for other processors
        mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, False)
        # Any add or insertion that would generate a new entry that has not
        # been preallocated will raise an error
        mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        # Do not ignore zeros while we fill the initial matrix so that
        # petsc doesn't compress things out.
        if not block_sparse:
            mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, False)
        # When zeroing rows (e.g. for enforcing Dirichlet bcs), keep those in
        # the nonzero structure of the matrix. Otherwise PETSc would compact
        # the sparsity and render our sparsity caching useless.
        mat.setOption(mat.Option.KEEP_NONZERO_PATTERN, True)
        # We completely fill the allocated matrix when zeroing the
        # entries, so raise an error if we "missed" one.
        mat.setOption(mat.Option.UNUSED_NONZERO_LOCATION_ERR, True)
        # Put zeros in all the places we might eventually put a value.
        with timed_region("MatZeroInitial"):
            sparsity.fill_with_zeros(mat, self.sparsity.dims[0][0],
                                     self.sparsity.maps, self.sparsity.iteration_regions,
                                     set_diag=self.sparsity._has_diagonal)
        mat.assemble()
        mat.setOption(mat.Option.NEW_NONZERO_LOCATION_ERR, True)
        # Now we've filled up our matrix, so the sparsity is
        # "complete", we can ignore subsequent zero entries.
        if not block_sparse:
            mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)
        self.handle = mat

    def _init_global_block(self):
        """Initialise this block in the case where the matrix maps either
        to or from a :class:`Global`"""

        if (isinstance(self.sparsity._dsets[0], GlobalDataSet) and isinstance(self.sparsity._dsets[1], GlobalDataSet)):
            # In this case both row and column are a Global.
            mat = _GlobalMat(comm=self.comm)
        else:
            mat = _DatMat(self.sparsity)
        self.handle = mat

    def __call__(self, access, path, lgmaps=None, unroll_map=False):
        """Override the parent __call__ method in order to special-case global
        blocks in matrices."""
        # One of the path entries was not an Arg.
        if path == (None, None):
            lgmaps, = lgmaps
            assert all(l is None for l in lgmaps)
            return _make_object('Arg',
                                data=self.handle.getPythonContext().global_,
                                access=access)
        elif None in path:
            thispath = path[0] or path[1]
            return _make_object('Arg', data=self.handle.getPythonContext().dat,
                                map=thispath, access=access)
        else:
            return super().__call__(access, path, lgmaps=lgmaps, unroll_map=unroll_map)

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
        yield from itertools.chain(*self.blocks)

    @collective
    def zero(self):
        """Zero the matrix."""
        self.assemble()
        self.handle.zeroEntries()

    @collective
    def zero_rows(self, rows, diag_val=1.0):
        """Zeroes the specified rows of the matrix, with the exception of the
        diagonal entry, which is set to diag_val. May be used for applying
        strong boundary conditions.

        :param rows: a :class:`Subset` or an iterable"""
        self.assemble()
        rows = rows.indices if isinstance(rows, Subset) else rows
        self.handle.zeroRowsLocal(rows, diag_val)

    def _flush_assembly(self):
        self.handle.assemble(assembly=PETSc.Mat.AssemblyType.FLUSH)

    @collective
    def set_local_diagonal_entries(self, rows, diag_val=1.0, idx=None):
        """Set the diagonal entry in ``rows`` to a particular value.

        :param rows: a :class:`Subset` or an iterable.
        :param diag_val: the value to add

        The indices in ``rows`` should index the process-local rows of
        the matrix (no mapping to global indexes is applied).
        """
        rows = np.asarray(rows, dtype=IntType)
        rbs, _ = self.dims[0][0]
        if rbs > 1:
            if idx is not None:
                rows = rbs * rows + idx
            else:
                rows = np.dstack([rbs*rows + i for i in range(rbs)]).flatten()
        rows = rows.reshape(-1, 1)
        self.change_assembly_state(Mat.INSERT_VALUES)
        if len(rows) > 0:
            values = np.full(rows.shape, diag_val, dtype=ScalarType)
            self.handle.setValuesLocalRCV(rows, rows, values,
                                          addv=PETSc.InsertMode.INSERT_VALUES)

    @collective
    def assemble(self):
        # If the matrix is nested, we need to check each subblock to
        # see if it needs assembling.  But if it's monolithic then the
        # subblock assembly doesn't do anything, so we don't do that.
        if self.sparsity.nested:
            self.handle.assemble()
            for m in self:
                if m.assembly_state != Mat.ASSEMBLED:
                    m.change_assembly_state(Mat.ASSEMBLED)
        else:
            # Instead, we assemble the full monolithic matrix.
            self.handle.assemble()
            for m in self:
                m.handle.assemble()
            self.change_assembly_state(Mat.ASSEMBLED)

    def addto_values(self, rows, cols, values):
        """Add a block of values to the :class:`Mat`."""
        self.change_assembly_state(Mat.ADD_VALUES)
        if len(values) > 0:
            self.handle.setValuesBlockedLocal(rows, cols, values,
                                              addv=PETSc.InsertMode.ADD_VALUES)

    def set_values(self, rows, cols, values):
        """Set a block of values in the :class:`Mat`."""
        self.change_assembly_state(Mat.INSERT_VALUES)
        if len(values) > 0:
            self.handle.setValuesBlockedLocal(rows, cols, values,
                                              addv=PETSc.InsertMode.INSERT_VALUES)

    @utils.cached_property
    def blocks(self):
        """2-dimensional array of matrix blocks."""
        return self._blocks

    @property
    def values(self):
        self.assemble()
        if self.nrows * self.ncols > 1000000:
            raise ValueError("Printing dense matrix with more than 1 million entries not allowed.\n"
                             "Are you sure you wanted to do this?")
        if (isinstance(self.sparsity._dsets[0], GlobalDataSet) or isinstance(self.sparsity._dsets[1], GlobalDataSet)):
            return self.handle.getPythonContext()[:, :]
        else:
            return self.handle[:, :]


class ParLoop(base.ParLoop):

    def log_flops(self, flops):
        PETSc.Log.logFlops(flops)


def _DatMat(sparsity, dat=None):
    """A :class:`PETSc.Mat` with global size nx1 or nx1 implemented as a
    :class:`.Dat`"""
    if isinstance(sparsity.dsets[0], GlobalDataSet):
        dset = sparsity.dsets[1]
        sizes = ((None, 1), (dset.size*dset.cdim, None))
    elif isinstance(sparsity.dsets[1], GlobalDataSet):
        dset = sparsity.dsets[0]
        sizes = ((dset.size * dset.cdim, None), (None, 1))
    else:
        raise ValueError("Not a DatMat")

    A = PETSc.Mat().createPython(sizes, comm=sparsity.comm)
    A.setPythonContext(_DatMatPayload(sparsity, dat))
    A.setUp()
    return A


class _DatMatPayload(object):

    def __init__(self, sparsity, dat=None, dset=None):
        if isinstance(sparsity.dsets[0], GlobalDataSet):
            self.dset = sparsity.dsets[1]
            self.sizes = ((None, 1), (self.dset.size * self.dset.cdim, None))
        elif isinstance(sparsity.dsets[1], GlobalDataSet):
            self.dset = sparsity.dsets[0]
            self.sizes = ((self.dset.size * self.dset.cdim, None), (None, 1))
        else:
            raise ValueError("Not a DatMat")

        self.sparsity = sparsity
        self.dat = dat or _make_object("Dat", self.dset, dtype=PETSc.ScalarType)
        self.dset = dset

    def __getitem__(self, key):
        shape = [s[0] or 1 for s in self.sizes]
        return self.dat.data_ro.reshape(*shape)[key]

    def zeroEntries(self, mat):
        self.dat.data[...] = 0.0

    def mult(self, mat, x, y):
        '''Y = mat x'''
        with self.dat.vec_ro as v:
            if self.sizes[0][0] is None:
                # Row matrix
                out = v.dot(x)
                if y.comm.rank == 0:
                    y.array[0] = out
                else:
                    y.array[...]
            else:
                # Column matrix
                if x.sizes[1] == 1:
                    v.copy(y)
                    a = np.zeros(1)
                    if x.comm.rank == 0:
                        a[0] = x.array_r
                    else:
                        x.array_r
                    x.comm.tompi4py().bcast(a)
                    return y.scale(a)
                else:
                    return v.pointwiseMult(x, y)

    def multTranspose(self, mat, x, y):
        with self.dat.vec_ro as v:
            if self.sizes[0][0] is None:
                # Row matrix
                if x.sizes[1] == 1:
                    v.copy(y)
                    a = np.zeros(1)
                    if x.comm.rank == 0:
                        a[0] = x.array_r
                    else:
                        x.array_r
                    x.comm.tompi4py().bcast(a)
                    y.scale(a)
                else:
                    v.pointwiseMult(x, y)
            else:
                # Column matrix
                out = v.dot(x)
                if y.comm.rank == 0:
                    y.array[0] = out
                else:
                    y.array[...]

    def multTransposeAdd(self, mat, x, y, z):
        ''' z = y + mat^Tx '''
        with self.dat.vec_ro as v:
            if self.sizes[0][0] is None:
                # Row matrix
                if x.sizes[1] == 1:
                    v.copy(z)
                    a = np.zeros(1)
                    if x.comm.rank == 0:
                        a[0] = x.array_r
                    else:
                        x.array_r
                    x.comm.tompi4py().bcast(a)
                    if y == z:
                        # Last two arguments are aliased.
                        tmp = y.duplicate()
                        y.copy(tmp)
                        y = tmp
                    z.scale(a)
                    z.axpy(1, y)
                else:
                    if y == z:
                        # Last two arguments are aliased.
                        tmp = y.duplicate()
                        y.copy(tmp)
                        y = tmp
                    v.pointwiseMult(x, z)
                    return z.axpy(1, y)
            else:
                # Column matrix
                out = v.dot(x)
                y = y.array_r
                if z.comm.rank == 0:
                    z.array[0] = out + y[0]
                else:
                    z.array[...]

    def duplicate(self, mat, copy=True):
        if copy:
            return _DatMat(self.sparsity, self.dat.duplicate())
        else:
            return _DatMat(self.sparsity)


def _GlobalMat(global_=None, comm=None):
    """A :class:`PETSc.Mat` with global size 1x1 implemented as a
    :class:`.Global`"""
    A = PETSc.Mat().createPython(((None, 1), (None, 1)), comm=comm)
    A.setPythonContext(_GlobalMatPayload(global_, comm))
    A.setUp()
    return A


class _GlobalMatPayload(object):

    def __init__(self, global_=None, comm=None):
        self.global_ = global_ or _make_object("Global", 1, dtype=PETSc.ScalarType, comm=comm)

    def __getitem__(self, key):
        return self.global_.data_ro.reshape(1, 1)[key]

    def zeroEntries(self, mat):
        self.global_.data[...] = 0.0

    def getDiagonal(self, mat, result=None):
        if result is None:
            result = self.global_.dataset.layout_vec.duplicate()
        if result.comm.rank == 0:
            result.array[...] = self.global_.data_ro
        else:
            result.array[...]
        return result

    def mult(self, mat, x, result):
        if result.comm.rank == 0:
            result.array[...] = self.global_.data_ro * x.array_r
        else:
            result.array[...]

    def multTransposeAdd(self, mat, x, y, z):
        if z.comm.rank == 0:
            ax = self.global_.data_ro * x.array_r
            if y == z:
                z.array[...] += ax
            else:
                z.array[...] = ax + y.array_r
        else:
            x.array_r
            y.array_r
            z.array[...]

    def duplicate(self, mat, copy=True):
        if copy:
            return _GlobalMat(self.global_.duplicate(), comm=mat.comm)
        else:
            return _GlobalMat(comm=mat.comm)
