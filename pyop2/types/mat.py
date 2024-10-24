import abc
import ctypes
import itertools
from collections.abc import Sequence

import numpy as np
from petsc4py import PETSc

from pyop2 import (
    caching,
    configuration as conf,
    datatypes as dtypes,
    exceptions as ex,
    mpi,
    profiling,
    sparsity,
    utils
)
from pyop2.types.access import Access
from pyop2.types.data_carrier import DataCarrier
from pyop2.types.dataset import DataSet, GlobalDataSet, MixedDataSet
from pyop2.types.map import Map, ComposedMap
from pyop2.types.set import MixedSet, Subset


class Sparsity(caching.ObjectCached):

    """OP2 Sparsity, the non-zero structure of a matrix derived from the block-wise specified pairs of :class:`Map` objects.

    Examples of constructing a Sparsity: ::

        Sparsity((row_dset, col_dset),
                 [(first_rowmap, first_colmap), (second_rowmap, second_colmap), None])

    .. _MatMPIAIJSetPreallocation: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html
    """

    def __init__(self, dsets, maps_and_regions, name=None, nest=None, block_sparse=None, diagonal_block=True):
        r"""
        :param dsets: :class:`DataSet`\s for the left and right function
            spaces this :class:`Sparsity` maps between
        :param maps_and_regions: `dict` to build the :class:`Sparsity` from.
            ``maps_and_regions`` must be keyed by the block index pair (i, j).
            ``maps_and_regions[(i, j)]`` must be a list of tuples of
            ``(rmap, cmap, iteration_regions)``, where ``rmap`` and ``cmap``
            is a pair of :class:`Map`\s specifying a row map and a column map,
            and ``iteration_regions`` represent regions that select subsets
            of extruded maps to iterate over. If the matrix only has a single
            block, one can altenatively pass the value ``maps_and_regions[(0, 0)]``.
        :param string name: user-defined label (optional)
        :param nest: Should the sparsity over mixed set be built as nested blocks?
        :param block_sparse: Should the sparsity for datasets with
            cdim > 1 be built as a block sparsity?
        :param diagonal_block: Flag indicating whether this sparsity is for
            a matrix/submatrix located on the diagonal.
        """
        # Protect against re-initialization when retrieved from cache
        if self._initialized:
            return
        self._dsets = dsets
        self._maps_and_regions = maps_and_regions
        self._block_sparse = block_sparse
        self._diagonal_block = diagonal_block
        self.lcomm = mpi.internal_comm(self.dsets[0].comm, self)
        self.rcomm = mpi.internal_comm(self.dsets[1].comm, self)
        if isinstance(dsets[0], GlobalDataSet) or isinstance(dsets[1], GlobalDataSet):
            self._dims = (((1, 1),),)
            self._d_nnz = None
            self._o_nnz = None
        else:
            rset, cset = self.dsets
            self._has_diagonal = (rset == cset) and diagonal_block
            tmp = itertools.product([x.cdim for x in self.dsets[0]],
                                    [x.cdim for x in self.dsets[1]])
            dims = [[None for _ in range(self.shape[1])] for _ in range(self.shape[0])]
            for r in range(self.shape[0]):
                for c in range(self.shape[1]):
                    dims[r][c] = next(tmp)
            self._dims = tuple(tuple(d) for d in dims)
        if self.lcomm != self.rcomm:
            raise ValueError("Haven't thought hard enough about different left and right communicators")
        self.comm = mpi.internal_comm(self.lcomm, self)
        self._name = name or "sparsity_#x%x" % id(self)
        # If the Sparsity is defined on MixedDataSets, we need to build each
        # block separately
        if (isinstance(dsets[0], MixedDataSet) or isinstance(dsets[1], MixedDataSet)) \
           and nest:
            self._nested = True
            self._blocks = []
            for i, rds in enumerate(dsets[0]):
                row = []
                for j, cds in enumerate(dsets[1]):
                    row.append(Sparsity((rds, cds), tuple(self._maps_and_regions[(i, j)]) if (i, j) in self._maps_and_regions else (),
                                        block_sparse=block_sparse,
                                        diagonal_block=(dsets[0] is dsets[1] and i == j)))
                self._blocks.append(row)
            self._d_nnz = tuple(s._d_nnz for s in self)
            self._o_nnz = tuple(s._o_nnz for s in self)
        elif isinstance(dsets[0], GlobalDataSet) or isinstance(dsets[1], GlobalDataSet):
            # Where the sparsity maps either from or to a Global, we
            # don't really have any sparsity structure.
            self._blocks = [[self]]
            self._nested = False
        else:
            for dset in dsets:
                if isinstance(dset, MixedDataSet) and any([isinstance(d, GlobalDataSet) for d in dset]):
                    raise ex.SparsityFormatError("Mixed monolithic matrices with Global rows or columns are not supported.")
            self._nested = False
            with profiling.timed_region("CreateSparsity"):
                nnz, onnz = sparsity.build_sparsity(self)
                self._d_nnz = nnz
                self._o_nnz = onnz
            self._blocks = [[self]]
        self._initialized = True

    _cache = {}

    @classmethod
    @utils.validate_type(('name', str, ex.NameTypeError))
    def _process_args(cls, dsets, maps_and_regions, name=None, nest=None, block_sparse=None, diagonal_block=True):
        from pyop2.types import IterationRegion

        if len(dsets) != 2:
            raise RuntimeError(f"dsets must be a tuple of two DataSets: got {dsets}")
        for dset in dsets:
            if not isinstance(dset, DataSet) and dset is not None:
                raise ex.DataSetTypeError("All data sets must be of type DataSet, not type %r" % type(dset))
        if isinstance(maps_and_regions, Sequence):
            # Convert short-hand notation to generic one.
            maps_and_regions = {(0, 0): maps_and_regions}
        elif not isinstance(maps_and_regions, dict):
            raise TypeError(f"maps_and_regions must be dict or Sequence: got {type(maps_and_regions)}")
        processed_maps_and_regions = {(i, j): frozenset() for i, _ in enumerate(dsets[0]) for j, _ in enumerate(dsets[1])}
        for key, val in maps_and_regions.items():
            i, j = key  # block indices: (0, 0) if not mixed
            if i >= len(dsets[0]) or j >= len(dsets[1]):
                raise RuntimeError(f"(i, j) must be < {(len(dsets[0]), len(dsets[1]))}: got {(i, j)}")
            processed_val = set()
            for rmap, cmap, iteration_regions in set(val):
                if not isinstance(dsets[0][i], GlobalDataSet) and not isinstance(dsets[1][j], GlobalDataSet):
                    for m in [rmap, cmap]:
                        if not isinstance(m, Map):
                            raise ex.MapTypeError(
                                "All maps must be of type map, not type %r" % type(m))
                        if not isinstance(m, ComposedMap) and len(m.values_with_halo) == 0 and m.iterset.total_size > 0:
                            raise ex.MapValueError(
                                "Unpopulated map values when trying to build sparsity.")
                    if rmap.toset is not dsets[0][i].set or cmap.toset is not dsets[1][j].set:
                        raise RuntimeError("Map toset must be the same as DataSet set")
                    if rmap.iterset is not cmap.iterset:
                        raise RuntimeError("Iterset of both maps in a pair must be the same")
                if iteration_regions is None:
                    iteration_regions = (IterationRegion.ALL, )
                else:
                    iteration_regions = tuple(sorted(iteration_regions))
                processed_val.update(((rmap, cmap, iteration_regions), ))
            if len(processed_val) > 0:
                processed_maps_and_regions[key] = frozenset(processed_val)
        processed_maps_and_regions = dict(sorted(processed_maps_and_regions.items()))
        # Need to return the caching object, a tuple of the processed
        # arguments and a dict of kwargs.
        if isinstance(dsets[0], GlobalDataSet):
            cache = None
        elif isinstance(dsets[0].set, MixedSet):
            cache = dsets[0].set[0]
        else:
            cache = dsets[0].set
        if nest is None:
            nest = conf.configuration["matnest"]
        if block_sparse is None:
            block_sparse = conf.configuration["block_sparsity"]
        kwargs = {"name": name,
                  "nest": nest,
                  "block_sparse": block_sparse,
                  "diagonal_block": diagonal_block}
        return (cache,) + (tuple(dsets), processed_maps_and_regions), kwargs

    @classmethod
    def _cache_key(cls, dsets, maps_and_regions, name, nest, block_sparse, diagonal_block, *args, **kwargs):
        return (dsets, tuple(maps_and_regions.items()), nest, block_sparse)

    def __getitem__(self, idx):
        """Return :class:`Sparsity` block with row and column given by ``idx``
        or a given row of blocks."""
        try:
            i, j = idx
            return self._blocks[i][j]
        except TypeError:
            return self._blocks[idx]

    @utils.cached_property
    def dsets(self):
        r"""A pair of :class:`DataSet`\s for the left and right function
        spaces this :class:`Sparsity` maps between."""
        return self._dsets

    @utils.cached_property
    def rcmaps(self):
        return {key: [(_rmap, _cmap) for _rmap, _cmap, _ in val] for key, val in self._maps_and_regions.items()}

    @utils.cached_property
    def iteration_regions(self):
        return {key: [_iteration_regions for _, _, _iteration_regions in val] for key, val in self._maps_and_regions.items()}

    @utils.cached_property
    def dims(self):
        """A tuple of tuples where the ``i,j``th entry
        is a pair giving the number of rows per entry of the row
        :class:`Set` and the number of columns per entry of the column
        :class:`Set` of the ``Sparsity``.  The extents of the first
        two indices are given by the :attr:`shape` of the sparsity.
        """
        return self._dims

    @utils.cached_property
    def shape(self):
        """Number of block rows and columns."""
        return (len(self._dsets[0] or [1]),
                len(self._dsets[1] or [1]))

    @utils.cached_property
    def nested(self):
        r"""Whether a sparsity is monolithic (even if it has a block structure).

        To elaborate, if a sparsity maps between
        :class:`MixedDataSet`\s, it can either be nested, in which
        case it consists of as many blocks are the product of the
        length of the datasets it maps between, or monolithic.  In the
        latter case the sparsity is for the full map between the mixed
        datasets, rather than between the blocks of the non-mixed
        datasets underneath them.
        """
        return self._nested

    @utils.cached_property
    def name(self):
        """A user-defined label."""
        return self._name

    def __iter__(self):
        r"""Iterate over all :class:`Sparsity`\s by row and then by column."""
        for row in self._blocks:
            for s in row:
                yield s

    def __str__(self):
        return "OP2 Sparsity: dsets %s, maps_and_regions %s, name %s, nested %s, block_sparse %s, diagonal_block %s" % \
               (self._dsets, self._maps_and_regions, self._name, self._nested, self._block_sparse, self._diagonal_block)

    def __repr__(self):
        return "Sparsity(%r, %r, name=%r, nested=%r, block_sparse=%r, diagonal_block=%r)" % (self.dsets, self._maps_and_regions, self.name, self._nested, self._block_sparse, self._diagonal_block)

    @utils.cached_property
    def nnz(self):
        """Array containing the number of non-zeroes in the various rows of the
        diagonal portion of the local submatrix.

        This is the same as the parameter `d_nnz` used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return self._d_nnz

    @utils.cached_property
    def onnz(self):
        """Array containing the number of non-zeroes in the various rows of the
        off-diagonal portion of the local submatrix.

        This is the same as the parameter `o_nnz` used for preallocation in
        PETSc's MatMPIAIJSetPreallocation_."""
        return self._o_nnz

    @utils.cached_property
    def nz(self):
        return self._d_nnz.sum()

    @utils.cached_property
    def onz(self):
        return self._o_nnz.sum()

    def __contains__(self, other):
        """Return true if other is a pair of maps in self.maps(). This
        will also return true if the elements of other have parents in
        self.maps()."""
        for i, rm in enumerate(other[0]):
            for j, cm in enumerate(other[1]):
                for maps in self.rcmaps[(i, j)]:
                    if (rm, cm) <= maps:
                        break
                else:
                    return False
        return True


class SparsityBlock(Sparsity):
    """A proxy class for a block in a monolithic :class:`.Sparsity`.

    :arg parent: The parent monolithic sparsity.
    :arg i: The block row.
    :arg j: The block column.

    .. warning::

       This class only implements the properties necessary to infer
       its shape.  It does not provide arrays of non zero fill."""
    def __init__(self, parent, i, j):
        # Protect against re-initialization when retrieved from cache
        if self._initialized:
            return

        self._dsets = (parent.dsets[0][i], parent.dsets[1][j])
        self._maps_and_regions = {(0, 0): tuple(parent._maps_and_regions[(i, j)]) if (i, j) in parent._maps_and_regions else ()}
        self._has_diagonal = i == j and parent._has_diagonal
        self._parent = parent
        self._dims = tuple([tuple([parent.dims[i][j]])])
        self._blocks = [[self]]
        self.lcomm = mpi.internal_comm(self.dsets[0].comm, self)
        self.rcomm = mpi.internal_comm(self.dsets[1].comm, self)
        # TODO: think about lcomm != rcomm
        self.comm = mpi.internal_comm(self.lcomm, self)
        self._initialized = True

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


class AbstractMat(DataCarrier, abc.ABC):
    r"""OP2 matrix data. A ``Mat`` is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`.

    When a ``Mat`` is passed to :func:`pyop2.op2.par_loop`, the maps via which
    indirection occurs for the row and column space, and the access
    descriptor are passed by `calling` the ``Mat``. For instance, if a
    ``Mat`` named ``A`` is to be accessed for reading via a row :class:`Map`
    named ``R`` and a column :class:`Map` named ``C``, this is accomplished by::

     A(pyop2.READ, (R[pyop2.i[0]], C[pyop2.i[1]]))

    Notice that it is `always` necessary to index the indirection maps
    for a ``Mat``. See the :class:`Mat` documentation for more
    details.

    .. note ::

       After executing :func:`par_loop`\s that write to a ``Mat`` and
       before using it (for example to view its values), you must call
       :meth:`assemble` to finalise the writes.
    """

    ASSEMBLED = "ASSEMBLED"
    INSERT_VALUES = "INSERT_VALUES"
    ADD_VALUES = "ADD_VALUES"

    _modes = [Access.WRITE, Access.INC]

    @utils.validate_type(('sparsity', Sparsity, ex.SparsityTypeError),
                         ('name', str, ex.NameTypeError))
    def __init__(self, sparsity, dtype=None, name=None):
        self._sparsity = sparsity
        self.lcomm = mpi.internal_comm(sparsity.lcomm, self)
        self.rcomm = mpi.internal_comm(sparsity.rcomm, self)
        self.comm = mpi.internal_comm(sparsity.comm, self)
        dtype = dtype or dtypes.ScalarType
        self._datatype = np.dtype(dtype)
        self._name = name or "mat_#x%x" % id(self)
        self.assembly_state = Mat.ASSEMBLED

    @utils.validate_in(('access', _modes, ex.ModeValueError))
    def __call__(self, access, path, lgmaps=None, unroll_map=False):
        from pyop2.parloop import MatLegacyArg, MixedMatLegacyArg

        path_maps = utils.as_tuple(path, Map, 2)
        if conf.configuration["type_check"] and tuple(path_maps) not in self.sparsity:
            raise ex.MapValueError("Path maps not in sparsity maps")

        if self.is_mixed:
            return MixedMatLegacyArg(self, path, access, lgmaps, unroll_map)
        else:
            return MatLegacyArg(self, path, access, lgmaps, unroll_map)

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.dtype, self.dims)

    def assemble(self):
        """Finalise this :class:`Mat` ready for use.

        Call this /after/ executing all the par_loops that write to
        the matrix before you want to look at it.
        """
        raise NotImplementedError("Subclass should implement this")

    def addto_values(self, rows, cols, values):
        """Add a block of values to the :class:`Mat`."""
        raise NotImplementedError(
            "Abstract Mat base class doesn't know how to set values.")

    def set_values(self, rows, cols, values):
        """Set a block of values in the :class:`Mat`."""
        raise NotImplementedError(
            "Abstract Mat base class doesn't know how to set values.")

    @utils.cached_property
    def nblocks(self):
        return int(np.prod(self.sparsity.shape))

    @utils.cached_property
    def _argtypes_(self):
        """Ctypes argtype for this :class:`Mat`"""
        return tuple(ctypes.c_voidp for _ in self)

    @utils.cached_property
    def is_mixed(self):
        return self.sparsity.shape > (1, 1)

    @utils.cached_property
    def dims(self):
        """A pair of integers giving the number of matrix rows and columns for
        each member of the row :class:`Set` and column :class:`Set`
        respectively. This corresponds to the ``cdim`` member of a
        :class:`DataSet`."""
        return self._sparsity._dims

    @utils.cached_property
    def nrows(self):
        "The number of rows in the matrix (local to this process)"
        return self.sparsity.dsets[0].layout_vec.local_size

    @utils.cached_property
    def nblock_rows(self):
        """The number "block" rows in the matrix (local to this process).

        This is equivalent to the number of rows in the matrix divided
        by the dimension of the row :class:`DataSet`.
        """
        assert len(self.sparsity.dsets[0]) == 1, "Block rows don't make sense for mixed Mats"
        layout_vec = self.sparsity.dsets[0].layout_vec
        return layout_vec.local_size // layout_vec.block_size

    @utils.cached_property
    def nblock_cols(self):
        """The number of "block" columns in the matrix (local to this process).

        This is equivalent to the number of columns in the matrix
        divided by the dimension of the column :class:`DataSet`.
        """
        assert len(self.sparsity.dsets[1]) == 1, "Block cols don't make sense for mixed Mats"
        layout_vec = self.sparsity.dsets[1].layout_vec
        return layout_vec.local_size // layout_vec.block_size

    @utils.cached_property
    def ncols(self):
        "The number of columns in the matrix (local to this process)"
        return self.sparsity.dsets[1].layout_vec.local_size

    @utils.cached_property
    def sparsity(self):
        """:class:`Sparsity` on which the ``Mat`` is defined."""
        return self._sparsity

    @utils.cached_property
    def _is_scalar_field(self):
        # Sparsity from Dat to MixedDat has a shape like (1, (1, 1))
        # (which you can't take the product of)
        return all(np.prod(d) == 1 for d in self.dims)

    @utils.cached_property
    def _is_vector_field(self):
        return not self._is_scalar_field

    def change_assembly_state(self, new_state):
        """Switch the matrix assembly state."""
        if new_state == Mat.ASSEMBLED or self.assembly_state == Mat.ASSEMBLED:
            self.assembly_state = new_state
        elif new_state != self.assembly_state:
            self._flush_assembly()
            self.assembly_state = new_state
        else:
            pass

    def _flush_assembly(self):
        """Flush the in flight assembly operations (used when
        switching between inserting and adding values)."""
        pass

    @property
    def values(self):
        """A numpy array of matrix values.

        .. warning ::
            This is a dense array, so will need a lot of memory.  It's
            probably not a good idea to access this property if your
            matrix has more than around 10000 degrees of freedom.
        """
        raise NotImplementedError("Abstract base Mat does not implement values()")

    @utils.cached_property
    def dtype(self):
        """The Python type of the data."""
        return self._datatype

    @utils.cached_property
    def nbytes(self):
        """Return an estimate of the size of the data associated with this
        :class:`Mat` in bytes. This will be the correct size of the
        data payload, but does not take into account the (presumably
        small) overhead of the object and its metadata. The memory
        associated with the sparsity pattern is also not recorded.

        Note that this is the process local memory usage, not the sum
        over all MPI processes.
        """
        if self._sparsity._block_sparse:
            mult = np.sum(np.prod(self._sparsity.dims))
        else:
            mult = 1
        return (self._sparsity.nz + self._sparsity.onz) \
            * self.dtype.itemsize * mult

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __mul__(self, other):
        """Multiply this :class:`Mat` with the vector ``other``."""
        raise NotImplementedError("Abstract base Mat does not implement multiplication")

    def __str__(self):
        return "OP2 Mat: %s, sparsity (%s), datatype %s" \
               % (self._name, self._sparsity, self._datatype.name)

    def __repr__(self):
        return "Mat(%r, %r, %r)" \
               % (self._sparsity, self._datatype, self._name)


class Mat(AbstractMat):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the :class:`Sparsity`."""

    def __init__(self, *args, **kwargs):
        self.mat_type = kwargs.pop("mat_type", None)
        super().__init__(*args, **kwargs)
        self._init()
        self.assembly_state = Mat.ASSEMBLED

    # Firedrake relies on this to distinguish between MatBlock and not for boundary conditions
    local_to_global_maps = (None, None)

    @utils.cached_property
    def _kernel_args_(self):
        return tuple(a.handle.handle for a in self)

    @mpi.collective
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
        with profiling.timed_region("MatZeroInitial"):
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
        with profiling.timed_region("MatZeroInitial"):
            for i in range(rows):
                for j in range(cols):
                    sparsity.fill_with_zeros(self[i, j].handle,
                                             self[i, j].sparsity.dims[0][0],
                                             self[i, j].sparsity.rcmaps[(0, 0)],
                                             self[i, j].sparsity.iteration_regions[(0, 0)],
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
        with profiling.timed_region("MatZeroInitial"):
            sparsity.fill_with_zeros(mat, self.sparsity.dims[0][0],
                                     self.sparsity.rcmaps[(0, 0)],
                                     self.sparsity.iteration_regions[(0, 0)],
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
        from pyop2.parloop import GlobalLegacyArg, DatLegacyArg

        if path == (None, None):
            lgmaps, = lgmaps
            assert all(l is None for l in lgmaps)
            return GlobalLegacyArg(self.handle.getPythonContext().global_, access)
        elif None in path:
            thispath = path[0] or path[1]
            return DatLegacyArg(self.handle.getPythonContext().dat, thispath, access)
        else:
            return super().__call__(access, path, lgmaps, unroll_map)

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

    @property
    def dat_version(self):
        if self.assembly_state != Mat.ASSEMBLED:
            raise RuntimeError("Should not ask for state counter if the matrix is not assembled.")
        return self.handle.stateGet()

    @mpi.collective
    def zero(self):
        """Zero the matrix."""
        self.assemble()
        self.handle.zeroEntries()

    @mpi.collective
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

    @mpi.collective
    def set_local_diagonal_entries(self, rows, diag_val=1.0, idx=None):
        """Set the diagonal entry in ``rows`` to a particular value.

        :param rows: a :class:`Subset` or an iterable.
        :param diag_val: the value to add

        The indices in ``rows`` should index the process-local rows of
        the matrix (no mapping to global indexes is applied).
        """
        rows = np.asarray(rows, dtype=dtypes.IntType)
        rbs, _ = self.dims[0][0]
        if rbs > 1:
            if idx is not None:
                rows = rbs * rows + idx
            else:
                rows = np.dstack([rbs*rows + i for i in range(rbs)]).flatten()
        rows = rows.reshape(-1, 1)
        self.change_assembly_state(Mat.INSERT_VALUES)
        if len(rows) > 0:
            values = np.full(rows.shape, diag_val, dtype=dtypes.ScalarType)
            self.handle.setValuesLocalRCV(rows, rows, values,
                                          addv=PETSc.InsertMode.INSERT_VALUES)

    @mpi.collective
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


class MatBlock(AbstractMat):
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
        self.comm = mpi.internal_comm(parent.comm, self)
        self.local_to_global_maps = self.handle.getLGMap()

    @property
    def dat_version(self):
        return self.handle.stateGet()

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
        rows = np.asarray(rows, dtype=dtypes.IntType)
        rbs, _ = self.dims[0][0]
        if rbs > 1:
            if idx is not None:
                rows = rbs * rows + idx
            else:
                rows = np.dstack([rbs*rows + i for i in range(rbs)]).flatten()
        rows = rows.reshape(-1, 1)
        self.change_assembly_state(Mat.INSERT_VALUES)
        if len(rows) > 0:
            values = np.full(rows.shape, diag_val, dtype=dtypes.ScalarType)
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


class _DatMatPayload:

    def __init__(self, sparsity, dat=None, dset=None):
        from pyop2.types.dat import Dat

        if isinstance(sparsity.dsets[0], GlobalDataSet):
            self.dset = sparsity.dsets[1]
            self.sizes = ((None, 1), (self.dset.size * self.dset.cdim, None))
        elif isinstance(sparsity.dsets[1], GlobalDataSet):
            self.dset = sparsity.dsets[0]
            self.sizes = ((self.dset.size * self.dset.cdim, None), (None, 1))
        else:
            raise ValueError("Not a DatMat")

        self.sparsity = sparsity
        self.dat = dat or Dat(self.dset, dtype=PETSc.ScalarType)
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
                    a = np.zeros(1, dtype=dtypes.ScalarType)
                    if x.comm.rank == 0:
                        a[0] = x.array_r
                    else:
                        x.array_r
                    with mpi.temp_internal_comm(x.comm) as comm:
                        comm.bcast(a)
                    return y.scale(a)
                else:
                    return v.pointwiseMult(x, y)

    def multTranspose(self, mat, x, y):
        with self.dat.vec_ro as v:
            if self.sizes[0][0] is None:
                # Row matrix
                if x.sizes[1] == 1:
                    v.copy(y)
                    a = np.zeros(1, dtype=dtypes.ScalarType)
                    if x.comm.rank == 0:
                        a[0] = x.array_r
                    else:
                        x.array_r
                    with mpi.temp_internal_comm(x.comm) as comm:
                        comm.bcast(a)
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
                    a = np.zeros(1, dtype=dtypes.ScalarType)
                    if x.comm.rank == 0:
                        a[0] = x.array_r
                    else:
                        x.array_r
                    with mpi.temp_internal_comm(x.comm) as comm:
                        comm.bcast(a)
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


class _GlobalMatPayload:

    def __init__(self, global_=None, comm=None):
        from pyop2.types.glob import Global
        self.global_ = global_ or Global(1, dtype=PETSc.ScalarType, comm=comm)

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
