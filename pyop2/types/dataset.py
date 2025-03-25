import numbers

import numpy as np
from petsc4py import PETSc

from pyop2 import (
    caching,
    datatypes as dtypes,
    exceptions as ex,
    mpi,
    utils
)
from pyop2.types.set import ExtrudedSet, GlobalSet, MixedSet, Set, Subset


class DataSet(caching.ObjectCached):
    """PyOP2 Data Set

    Set used in the op2.Dat structures to specify the dimension of the data.
    """

    @utils.validate_type(('iter_set', Set, ex.SetTypeError),
                         ('dim', (numbers.Integral, tuple, list), ex.DimTypeError),
                         ('name', str, ex.NameTypeError),
                         ('apply_local_global_filter', bool, ex.DataTypeError))
    def __init__(self, iter_set, dim=1, name=None, apply_local_global_filter=False):
        if isinstance(iter_set, ExtrudedSet):
            raise NotImplementedError("Not allowed!")
        if self._initialized:
            return
        if isinstance(iter_set, Subset):
            raise NotImplementedError("Deriving a DataSet from a Subset is unsupported")
        self.comm = mpi.internal_comm(iter_set.comm, self)
        self._set = iter_set
        self._dim = utils.as_tuple(dim, numbers.Integral)
        self._cdim = np.prod(self._dim).item()
        self._name = name or "dset_#x%x" % id(self)
        self._initialized = True
        self._apply_local_global_filter = apply_local_global_filter

    @classmethod
    def _process_args(cls, *args, **kwargs):
        return (args[0], ) + args, kwargs

    @classmethod
    def _cache_key(cls, iter_set, dim=1, name=None, apply_local_global_filter=False):
        return (iter_set, utils.as_tuple(dim, numbers.Integral))

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (type(self), self.dim, self._set._wrapper_cache_key_, self._apply_local_global_filter)

    def __getstate__(self):
        """Extract state to pickle."""
        return self.__dict__

    def __setstate__(self, d):
        """Restore from pickled state."""
        self.__dict__.update(d)

    # Look up any unspecified attributes on the _set.
    def __getattr__(self, name):
        """Returns a Set specific attribute."""
        value = getattr(self.set, name)
        return value

    def __getitem__(self, idx):
        """Allow index to return self"""
        assert idx == 0
        return self

    @utils.cached_property
    def dim(self):
        """The shape tuple of the values for each element of the set."""
        return self._dim

    @utils.cached_property
    def cdim(self):
        """The scalar number of values for each member of the set. This is
        the product of the dim tuple."""
        return self._cdim

    @utils.cached_property
    def name(self):
        """Returns the name of the data set."""
        return self._name

    @utils.cached_property
    def set(self):
        """Returns the parent set of the data set."""
        return self._set

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 DataSet: %s on set %s, with dim %s, %s" % \
            (self._name, self._set, self._dim, self._apply_local_global_filter)

    def __repr__(self):
        return "DataSet(%r, %r, %r, %r)" % (self._set, self._dim, self._name, self._apply_local_global_filter)

    def __contains__(self, dat):
        """Indicate whether a given Dat is compatible with this DataSet."""
        return dat.dataset == self

    @utils.cached_property
    def lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`DataSet`.
        """
        lgmap = PETSc.LGMap()
        if self.comm.size == 1 and self.halo is None:
            lgmap.create(indices=np.arange(self.size, dtype=dtypes.IntType),
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
            nlocal_rows += dset.layout_vec.local_size
        offset = self.comm.scan(nlocal_rows)
        offset -= nlocal_rows
        for dset in self:
            nrows = dset.layout_vec.local_size
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
        size = ((self.size - self.set.constrained_size) * self.cdim, None)
        vec.setSizes(size, bsize=self.cdim)
        vec.setUp()
        return vec

    @utils.cached_property
    def dm(self):
        dm = PETSc.DMShell().create(comm=self.comm)
        dm.setGlobalVector(self.layout_vec)
        return dm


class GlobalDataSet(DataSet):
    """A proxy :class:`DataSet` for use in a :class:`Sparsity` where the
    matrix has :class:`Global` rows or columns."""

    def __init__(self, global_):
        """
        :param global_: The :class:`Global` on which this object is based."""
        if self._initialized:
            return
        self._global = global_
        self.comm = mpi.internal_comm(global_.comm, self)
        self._globalset = GlobalSet(comm=self.comm)
        self._name = "gdset_#x%x" % id(self)
        self._initialized = True

    @classmethod
    def _cache_key(cls, *args):
        return None

    @utils.cached_property
    def dim(self):
        """The shape tuple of the values for each element of the set."""
        return self._global._dim

    @utils.cached_property
    def cdim(self):
        """The scalar number of values for each member of the set. This is
        the product of the dim tuple."""
        return self._global._cdim

    @utils.cached_property
    def name(self):
        """Returns the name of the data set."""
        return self._global._name

    @utils.cached_property
    def set(self):
        """Returns the parent set of the data set."""
        return self._globalset

    @utils.cached_property
    def size(self):
        """The number of local entries in the Dataset (1 on rank 0)"""
        return 1 if mpi.MPI.comm.rank == 0 else 0

    def __iter__(self):
        """Yield self when iterated over."""
        yield self

    def __len__(self):
        """This is not a mixed type and therefore of length 1."""
        return 1

    def __str__(self):
        return "OP2 GlobalDataSet: %s on Global %s" % \
            (self._name, self._global)

    def __repr__(self):
        return "GlobalDataSet(%r)" % (self._global)

    @utils.cached_property
    def lgmap(self):
        """A PETSc LGMap mapping process-local indices to global
        indices for this :class:`DataSet`.
        """
        lgmap = PETSc.LGMap()
        lgmap.create(indices=np.arange(1, dtype=dtypes.IntType),
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


class MixedDataSet(DataSet):
    r"""A container for a bag of :class:`DataSet`\s.

    Initialized either from a :class:`MixedSet` and an iterable or iterator of
    ``dims`` of corresponding length ::

        mdset = op2.MixedDataSet(mset, [dim1, ..., dimN])

    or from a tuple of :class:`Set`\s and an iterable of ``dims`` of
    corresponding length ::

        mdset = op2.MixedDataSet([set1, ..., setN], [dim1, ..., dimN])

    If all ``dims`` are to be the same, they can also be given as an
    :class:`int` for either of above invocations ::

        mdset = op2.MixedDataSet(mset, dim)
        mdset = op2.MixedDataSet([set1, ..., setN], dim)

    Initialized from a :class:`MixedSet` without explicitly specifying ``dims``
    they default to 1 ::

        mdset = op2.MixedDataSet(mset)

    Initialized from an iterable or iterator of :class:`DataSet`\s and/or
    :class:`Set`\s, where :class:`Set`\s are implicitly upcast to
    :class:`DataSet`\s of dim 1 ::

        mdset = op2.MixedDataSet([dset1, ..., dsetN])
    """

    def __init__(self, arg, dims=None):
        r"""
        :param arg:  a :class:`MixedSet` or an iterable or a generator
                     expression of :class:`Set`\s or :class:`DataSet`\s or a
                     mixture of both
        :param dims: `None` (the default) or an :class:`int` or an iterable or
                     generator expression of :class:`int`\s, which **must** be
                     of same length as `arg`

        .. Warning ::
            When using generator expressions for ``arg`` or ``dims``, these
            **must** terminate or else will cause an infinite loop.
        """
        if self._initialized:
            return
        self._dsets = arg
        try:
            # Try to choose the comm to be the same as the first set
            # of the MixedDataSet
            comm = self._process_args(arg, dims)[0][0].comm
        except AttributeError:
            comm = None
        self.comm = mpi.internal_comm(comm, self)
        self._initialized = True

    @classmethod
    def _process_args(cls, arg, dims=None):
        # If the second argument is not None it is expect to be a scalar dim
        # or an iterable of dims and the first is expected to be a MixedSet or
        # an iterable of Sets
        if dims is not None:
            # If arg is a MixedSet, get its Sets tuple
            sets = arg.split if isinstance(arg, MixedSet) else tuple(arg)
            # If dims is a scalar, turn it into a tuple of right length
            dims = (dims,) * len(sets) if isinstance(dims, int) else tuple(dims)
            if len(sets) != len(dims):
                raise ValueError("Got MixedSet of %d Sets but %s dims" %
                                 (len(sets), len(dims)))
            dsets = tuple(s ** d for s, d in zip(sets, dims))
        # Otherwise expect the first argument to be an iterable of Sets and/or
        # DataSets and upcast Sets to DataSets as necessary
        else:
            arg = [s if isinstance(s, DataSet) else s ** 1 for s in arg]
            dsets = utils.as_tuple(arg, type=DataSet)

        return (dsets[0].set, ) + (dsets, ), {}

    @classmethod
    def _cache_key(cls, arg, dims=None):
        return arg

    @utils.cached_property
    def _wrapper_cache_key_(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        """Return :class:`DataSet` with index ``idx`` or a given slice of datasets."""
        return self._dsets[idx]

    @utils.cached_property
    def split(self):
        r"""The underlying tuple of :class:`DataSet`\s."""
        return self._dsets

    @utils.cached_property
    def dim(self):
        """The shape tuple of the values for each element of the sets."""
        return tuple(s.dim for s in self._dsets)

    @utils.cached_property
    def cdim(self):
        """The sum of the scalar number of values for each member of the sets.
        This is the sum of products of the dim tuples."""
        return sum(s.cdim for s in self._dsets)

    @utils.cached_property
    def name(self):
        """Returns the name of the data sets."""
        return tuple(s.name for s in self._dsets)

    @utils.cached_property
    def set(self):
        """Returns the :class:`MixedSet` this :class:`MixedDataSet` is
        defined on."""
        return MixedSet(s.set for s in self._dsets)

    def __iter__(self):
        r"""Yield all :class:`DataSet`\s when iterated over."""
        for ds in self._dsets:
            yield ds

    def __len__(self):
        """Return number of contained :class:`DataSet`s."""
        return len(self._dsets)

    def __str__(self):
        return "OP2 MixedDataSet composed of DataSets: %s" % (self._dsets,)

    def __repr__(self):
        return "MixedDataSet(%r)" % (self._dsets,)

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
        if self.comm.size == 1 and self.halo is None:
            size = sum((s.size - s.constrained_size) * s.cdim for s in self)
            lgmap.create(indices=np.arange(size, dtype=dtypes.IntType),
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
        indices = np.full(idx_size, -1, dtype=dtypes.IntType)
        owned_sz = np.array([sum((s.size - s.constrained_size) * s.cdim for s in self)],
                            dtype=dtypes.IntType)
        field_offset = np.empty_like(owned_sz)
        self.comm.Scan(owned_sz, field_offset)
        field_offset -= owned_sz

        all_field_offsets = np.empty(self.comm.size, dtype=dtypes.IntType)
        self.comm.Allgather(field_offset, all_field_offsets)

        start = 0
        all_local_offsets = np.zeros(self.comm.size, dtype=dtypes.IntType)
        current_offsets = np.zeros(self.comm.size + 1, dtype=dtypes.IntType)
        for s in self:
            idx = indices[start:start + s.total_size * s.cdim]
            owned_sz[0] = (s.size - s.set.constrained_size) * s.cdim
            self.comm.Scan(owned_sz, field_offset)
            self.comm.Allgather(field_offset, current_offsets[1:])
            # Find the ranks each entry in the l2g belongs to
            l2g = s.unblocked_lgmap.indices
            tmp_indices = np.searchsorted(current_offsets, l2g, side="right") - 1
            idx[:] = l2g[:] - current_offsets[tmp_indices] + \
                all_field_offsets[tmp_indices] + all_local_offsets[tmp_indices]
            # Explicitly set -1 for constrained DoFs.
            idx[l2g < 0] = -1
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
