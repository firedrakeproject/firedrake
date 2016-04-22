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
import numpy as np

import base
from base import *
from logger import debug, warning
from versioning import CopyOnWrite, modifies, zeroes
from profiling import timed_region
import mpi
from mpi import collective
import sparsity
from pyop2 import utils


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


class DataSet(base.DataSet):

    @utils.cached_property
    def lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`DataSet`.
        """
        lgmap = PETSc.LGMap()
        if MPI.comm.size == 1:
            lgmap.create(indices=np.arange(self.size, dtype=PETSc.IntType),
                         bsize=self.cdim)
        else:
            lgmap.create(indices=self.halo.global_to_petsc_numbering,
                         bsize=self.cdim)
        return lgmap

    @utils.cached_property
    def unblocked_lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`DataSet` with a block size of 1.
        """
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
        offset = mpi.MPI.comm.scan(nlocal_rows)
        offset -= nlocal_rows
        for dset in self:
            nrows = dset.size * dset.cdim
            iset = PETSc.IS().createStride(nrows, first=offset, step=1)
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
            iset = PETSc.IS().createStride(n, first=start, step=1)
            iset.setBlockSize(bs)
            start += n
            ises.append(iset)
        return tuple(ises)

    @utils.cached_property
    def layout_vec(self):
        """A PETSc Vec compatible with the dof layout of this DataSet."""
        vec = PETSc.Vec().create()
        size = (self.size * self.cdim, None)
        vec.setSizes(size, bsize=self.cdim)
        vec.setUp()
        return vec


class MixedDataSet(DataSet, base.MixedDataSet):

    @utils.cached_property
    def layout_vec(self):
        """A PETSc Vec compatible with the dof layout of this MixedDataSet."""
        vec = PETSc.Vec().create()
        # Size of flattened vector is product of size and cdim of each dat
        size = sum(d.size * d.cdim for d in self)
        vec.setSizes((size, None))
        vec.setUp()
        return vec

    @utils.cached_property
    def vecscatters(self):
        """Get the vecscatters from the dof layout of this dataset to a PETSc Vec."""
        # To be compatible with a MatNest (from a MixedMat) the
        # ordering of a MixedDat constructed of Dats (x_0, ..., x_k)
        # on P processes is:
        # (x_0_0, x_1_0, ..., x_k_0, x_0_1, x_1_1, ..., x_k_1, ..., x_k_P)
        # That is, all the Dats from rank 0, followed by those of
        # rank 1, ...
        # Hence the offset into the global Vec is the exclusive
        # prefix sum of the local size of the mixed dat.
        size = sum(d.size * d.cdim for d in self)
        offset = MPI.comm.exscan(size)
        if offset is None:
            offset = 0
        scatters = []
        for d in self:
            size = d.size * d.cdim
            vscat = PETSc.Scatter().create(d.layout_vec, None, self.layout_vec,
                                           PETSc.IS().createStride(size, offset, 1))
            offset += size
            scatters.append(vscat)
        return tuple(scatters)

    @utils.cached_property
    def lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`MixedDataSet`.
        """
        lgmap = PETSc.LGMap()
        if MPI.comm.size == 1:
            size = sum(s.size * s.cdim for s in self)
            lgmap.create(indices=np.arange(size, dtype=PETSc.IntType),
                         bsize=1)
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
        indices = np.full(idx_size, -1, dtype=PETSc.IntType)
        owned_sz = np.array([sum(s.size * s.cdim for s in self)], dtype=PETSc.IntType)
        field_offset = np.empty_like(owned_sz)
        MPI.comm.Scan(owned_sz, field_offset)
        field_offset -= owned_sz

        all_field_offsets = np.empty(MPI.comm.size, dtype=PETSc.IntType)
        MPI.comm.Allgather(field_offset, all_field_offsets)

        start = 0
        all_local_offsets = np.zeros(MPI.comm.size, dtype=PETSc.IntType)
        current_offsets = np.zeros(MPI.comm.size + 1, dtype=PETSc.IntType)
        for s in self:
            idx = indices[start:start + s.total_size * s.cdim]
            owned_sz[0] = s.size * s.cdim
            MPI.comm.Scan(owned_sz, field_offset)
            MPI.comm.Allgather(field_offset, current_offsets[1:])
            # Find the ranks each entry in the l2g belongs to
            l2g = s.halo.global_to_petsc_numbering
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
            MPI.comm.Allgather(owned_sz, current_offsets[1:])
            all_local_offsets += current_offsets[1:]
            start += s.total_size * s.cdim
        lgmap.create(indices=indices, bsize=1)
        return lgmap

    @utils.cached_property
    def unblocked_lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`DataSet` with a block size of 1.
        """
        return self.lgmap


class Dat(base.Dat):

    @contextmanager
    def vec_context(self, readonly=True):
        """A context manager for a :class:`PETSc.Vec` from a :class:`Dat`.

        :param readonly: Access the data read-only (use :meth:`Dat.data_ro`)
                         or read-write (use :meth:`Dat.data`). Read-write
                         access requires a halo update."""

        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        acc = (lambda d: d.data_ro) if readonly else (lambda d: d.data)
        # Getting the Vec needs to ensure we've done all current computation.
        # If we only want readonly access then there's no need to
        # force the evaluation of reads from the Dat.
        self._force_evaluation(read=True, write=not readonly)
        if not hasattr(self, '_vec'):
            # Can't duplicate layout_vec of dataset, because we then
            # carry around extra unnecessary data.
            # But use getSizes to save an Allreduce in computing the
            # global size.
            size = self.dataset.layout_vec.getSizes()
            self._vec = PETSc.Vec().createWithArray(acc(self), size=size,
                                                    bsize=self.cdim)
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()
        yield self._vec
        if not readonly:
            self.needs_halo_update = True

    @property
    @modifies
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

        assert self.dtype == PETSc.ScalarType, \
            "Can't create Vec with type %s, must be %s" % (self.dtype, PETSc.ScalarType)
        acc = (lambda d: d.vec_ro) if readonly else (lambda d: d.vec)
        # Allocate memory for the contiguous vector
        if not hasattr(self, '_vec'):
            # In this case we can just duplicate the layout vec
            # because we're not placing an array.
            self._vec = self.dataset.layout_vec.duplicate()

        scatters = self.dataset.vecscatters
        # Do the actual forward scatter to fill the full vector with values
        for d, vscat in zip(self, scatters):
            with acc(d) as v:
                vscat.scatterBegin(v, self._vec, addv=PETSc.InsertMode.INSERT_VALUES)
                vscat.scatterEnd(v, self._vec, addv=PETSc.InsertMode.INSERT_VALUES)
        yield self._vec
        if not readonly:
            # Reverse scatter to get the values back to their original locations
            for d, vscat in zip(self, scatters):
                with acc(d) as v:
                    vscat.scatterBegin(self._vec, v, addv=PETSc.InsertMode.INSERT_VALUES,
                                       mode=PETSc.ScatterMode.REVERSE)
                    vscat.scatterEnd(self._vec, v, addv=PETSc.InsertMode.INSERT_VALUES,
                                     mode=PETSc.ScatterMode.REVERSE)
            self.needs_halo_update = True

    @property
    @modifies
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
        self._parent = parent
        self._dims = tuple([tuple([parent.dims[i][j]])])
        self._blocks = [[self]]

    @classmethod
    def _process_args(cls, *args, **kwargs):
        return (None, ) + args, kwargs

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        return None

    def __repr__(self):
        return "SparsityBlock(%r, %r, %r)" % (self._parent, self._i, self._j)


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

    @property
    def assembly_state(self):
        # Track our assembly state only
        return self._parent.assembly_state

    @assembly_state.setter
    def assembly_state(self, state):
        # Need to update our state and our parent's
        self._parent.assembly_state = state

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
        rows = np.asarray(rows, dtype=PETSc.IntType)
        rbs, _ = self.dims[0][0]
        # No need to set anything if we didn't get any rows.
        if len(rows) == 0:
            return
        if rbs > 1:
            if idx is not None:
                rows = rbs * rows + idx
            else:
                rows = np.dstack([rbs*rows + i for i in range(rbs)]).flatten()
        vals = np.repeat(diag_val, len(rows))
        closure = lambda: self.handle.setValuesLocalRCV(rows.reshape(-1, 1),
                                                        rows.reshape(-1, 1),
                                                        vals.reshape(-1, 1),
                                                        addv=PETSc.InsertMode.INSERT_VALUES)
        base._LazyMatOp(self, closure, new_state=Mat.INSERT_VALUES,
                        write=True).enqueue()

    def addto_values(self, rows, cols, values):
        """Add a block of values to the :class:`Mat`."""
        closure = lambda: self.handle.setValuesBlockedLocal(rows, cols, values,
                                                            addv=PETSc.InsertMode.ADD_VALUES)
        base._LazyMatOp(self, closure, new_state=Mat.ADD_VALUES,
                        read=True, write=True).enqueue()

    def set_values(self, rows, cols, values):
        """Set a block of values in the :class:`Mat`."""
        closure = lambda: self.handle.setValuesBlockedLocal(rows, cols, values,
                                                            addv=PETSc.InsertMode.INSERT_VALUES)
        base._LazyMatOp(self, closure, new_state=Mat.INSERT_VALUES,
                        write=True).enqueue()

    def assemble(self):
        raise RuntimeError("Should never call assemble on MatBlock")

    def _assemble(self):
        raise RuntimeError("Should never call _assemble on MatBlock")

    @property
    def values(self):
        rset, cset = self._parent.sparsity.dsets
        rowis = rset.field_ises[self._i]
        colis = cset.field_ises[self._j]
        base._trace.evaluate(set([self._parent]), set())
        self._parent.assemble()
        mat = self._parent.handle.getSubMatrix(isrow=rowis,
                                               iscol=colis)
        return mat[:, :]

    @property
    def dtype(self):
        return self._parent.dtype

    @property
    def nbytes(self):
        return self._parent.nbytes / (np.prod(self.sparsity.shape))

    def __repr__(self):
        return "MatBlock(%r, %r, %r)" % (self._parent, self._i, self._j)

    def __str__(self):
        return "Block[%s, %s] of %s" % (self._i, self._j, self._parent)


class Mat(base.Mat, CopyOnWrite):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    def __init__(self, *args, **kwargs):
        base.Mat.__init__(self, *args, **kwargs)
        CopyOnWrite.__init__(self, *args, **kwargs)
        self._init()
        self.assembly_state = Mat.ASSEMBLED

    @collective
    def _init(self):
        if not self.dtype == PETSc.ScalarType:
            raise RuntimeError("Can only create a matrix of type %s, %s is not supported"
                               % (PETSc.ScalarType, self.dtype))
        # If the Sparsity is defined on MixedDataSets, we need to build a MatNest
        if self.sparsity.shape > (1, 1):
            if self.sparsity.nested:
                self._init_nest()
                self._nested = True
            else:
                self._init_monolithic()
        else:
            self._init_block()

    def _init_monolithic(self):
        mat = PETSc.Mat()
        rset, cset = self.sparsity.dsets
        if rset.cdim != 1:
            rlgmap = rset.unblocked_lgmap
        else:
            rlgmap = rset.lgmap
        if cset.cdim != 1:
            clgmap = cset.unblocked_lgmap
        else:
            clgmap = cset.lgmap
        mat.createAIJ(size=((self.nrows, None), (self.ncols, None)),
                      nnz=(self.sparsity.nnz, self.sparsity.onnz),
                      bsize=1)
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
        mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, True)
        mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        # Put zeros in all the places we might eventually put a value.
        with timed_region("Zero initial matrix"):
            for i in range(rows):
                for j in range(cols):
                    sparsity.fill_with_zeros(self[i, j].handle,
                                             self[i, j].sparsity.dims[0][0],
                                             self[i, j].sparsity.maps,
                                             set_diag=(i == j))

        mat.assemble()
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
                       isrows=rset.field_ises, iscols=cset.field_ises)
        self.handle = mat

    def _init_block(self):
        self._blocks = [[self]]
        mat = PETSc.Mat()
        row_lg = self.sparsity.dsets[0].lgmap
        col_lg = self.sparsity.dsets[1].lgmap
        rdim, cdim = self.dims[0][0]

        if rdim == cdim and rdim > 1:
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
               bsize=(rdim, cdim))
        mat.setLGMap(rmap=row_lg, cmap=col_lg)
        # Do not stash entries destined for other processors, just drop them
        # (we take care of those in the halo)
        mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, True)
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
        with timed_region("Zero initial matrix"):
            sparsity.fill_with_zeros(mat, self.sparsity.dims[0][0], self.sparsity.maps)

        # Now we've filled up our matrix, so the sparsity is
        # "complete", we can ignore subsequent zero entries.
        if not block_sparse:
            mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)
        self.handle = mat
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

    @zeroes
    @collective
    def zero(self):
        """Zero the matrix."""
        base._trace.evaluate(set(), set([self]))
        self.handle.zeroEntries()

    @modifies
    @collective
    def zero_rows(self, rows, diag_val=1.0):
        """Zeroes the specified rows of the matrix, with the exception of the
        diagonal entry, which is set to diag_val. May be used for applying
        strong boundary conditions.

        :param rows: a :class:`Subset` or an iterable"""
        base._trace.evaluate(set([self]), set([self]))
        self._assemble()
        rows = rows.indices if isinstance(rows, Subset) else rows
        self.handle.zeroRowsLocal(rows, diag_val)

    def _cow_actual_copy(self, src):
        base._trace.evaluate(set([src]), set())
        self.handle = src.handle.duplicate(copy=True)
        return self

    def _flush_assembly(self):
        self.handle.assemble(assembly=PETSc.Mat.AssemblyType.FLUSH)

    @modifies
    @collective
    def set_local_diagonal_entries(self, rows, diag_val=1.0, idx=None):
        """Set the diagonal entry in ``rows`` to a particular value.

        :param rows: a :class:`Subset` or an iterable.
        :param diag_val: the value to add

        The indices in ``rows`` should index the process-local rows of
        the matrix (no mapping to global indexes is applied).
        """
        rows = np.asarray(rows, dtype=PETSc.IntType)
        rbs, _ = self.dims[0][0]
        # No need to set anything if we didn't get any rows.
        if len(rows) == 0:
            return
        if rbs > 1:
            if idx is not None:
                rows = rbs * rows + idx
            else:
                rows = np.dstack([rbs*rows + i for i in range(rbs)]).flatten()
        vals = np.repeat(diag_val, len(rows))
        closure = lambda: self.handle.setValuesLocalRCV(rows.reshape(-1, 1),
                                                        rows.reshape(-1, 1),
                                                        vals.reshape(-1, 1),
                                                        addv=PETSc.InsertMode.INSERT_VALUES)
        base._LazyMatOp(self, closure, new_state=Mat.INSERT_VALUES,
                        write=True).enqueue()

    @collective
    def _assemble(self):
        # If the matrix is nested, we need to check each subblock to
        # see if it needs assembling.  But if it's monolithic then the
        # subblock assembly doesn't do anything, so we don't do that.
        if self.sparsity.nested:
            for m in self:
                if m.assembly_state is not Mat.ASSEMBLED:
                    m.handle.assemble()
                m.assembly_state = Mat.ASSEMBLED
        # Instead, we assemble the full monolithic matrix.
        if self.assembly_state is not Mat.ASSEMBLED:
            self.handle.assemble()
            self.assembly_state = Mat.ASSEMBLED

    def addto_values(self, rows, cols, values):
        """Add a block of values to the :class:`Mat`."""
        closure = lambda: self.handle.setValuesBlockedLocal(rows, cols, values,
                                                            addv=PETSc.InsertMode.ADD_VALUES)
        base._LazyMatOp(self, closure, new_state=Mat.ADD_VALUES,
                        read=True, write=True).enqueue()

    def set_values(self, rows, cols, values):
        """Set a block of values in the :class:`Mat`."""
        closure = lambda: self.handle.setValuesBlockedLocal(rows, cols, values,
                                                            addv=PETSc.InsertMode.INSERT_VALUES)
        base._LazyMatOp(self, closure, new_state=Mat.INSERT_VALUES,
                        write=True).enqueue()

    @cached_property
    def blocks(self):
        """2-dimensional array of matrix blocks."""
        return self._blocks

    @property
    @modifies
    def values(self):
        base._trace.evaluate(set([self]), set())
        if self.nrows * self.ncols > 1000000:
            raise ValueError("Printing dense matrix with more than 1 million entries not allowed.\n"
                             "Are you sure you wanted to do this?")
        return self.handle[:, :]


class ParLoop(base.ParLoop):

    def log_flops(self):
        PETSc.Log.logFlops(self.num_flops)

# FIXME: Eventually (when we have a proper OpenCL solver) this wants to go in
# sequential


class Solver(base.Solver, PETSc.KSP):

    _cnt = 0

    def __init__(self, parameters=None, **kwargs):
        super(Solver, self).__init__(parameters, **kwargs)
        self._count = Solver._cnt
        Solver._cnt += 1
        self.create(PETSc.COMM_WORLD)
        self._opt_prefix = 'pyop2_ksp_%d' % self._count
        self.setOptionsPrefix(self._opt_prefix)
        converged_reason = self.ConvergedReason()
        self._reasons = dict([(getattr(converged_reason, r), r)
                              for r in dir(converged_reason)
                              if not r.startswith('_')])

    @collective
    def _set_parameters(self):
        opts = PETSc.Options(self._opt_prefix)
        for k, v in self.parameters.iteritems():
            if type(v) is bool:
                if v:
                    opts[k] = None
                else:
                    continue
            else:
                opts[k] = v
        self.setFromOptions()

    def __del__(self):
        # Remove stuff from the options database
        # It's fixed size, so if we don't it gets too big.
        if hasattr(self, '_opt_prefix'):
            opts = PETSc.Options()
            for k in self.parameters.iterkeys():
                del opts[self._opt_prefix + k]
            delattr(self, '_opt_prefix')

    @collective
    def _solve(self, A, x, b):
        self._set_parameters()
        # Set up the operator only if it has changed
        if not self.getOperators()[0] == A.handle:
            self.setOperators(A.handle)
            if self.parameters['pc_type'] == 'fieldsplit' and A.sparsity.shape != (1, 1):
                ises = A.sparsity.toset.field_ises
                fises = [(str(i), iset) for i, iset in enumerate(ises)]
                self.getPC().setFieldSplitIS(*fises)
        if self.parameters['plot_convergence']:
            self.reshist = []

            def monitor(ksp, its, norm):
                self.reshist.append(norm)
                debug("%3d KSP Residual norm %14.12e" % (its, norm))
            self.setMonitor(monitor)
        # Not using super here since the MRO would call base.Solver.solve
        with timed_region("PETSc Krylov solver"):
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
