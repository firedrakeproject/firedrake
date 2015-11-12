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

"""
User API Unit Tests
"""

import pytest
import numpy as np
from numpy.testing import assert_equal
from mpi4py import MPI

from pyop2 import op2
from pyop2 import exceptions
from pyop2 import sequential
from pyop2 import base


@pytest.fixture
def set():
    return op2.Set(5, 'foo')


@pytest.fixture
def iterset():
    return op2.Set(2, 'iterset')


@pytest.fixture
def toset():
    return op2.Set(3, 'toset')


@pytest.fixture
def sets(set, iterset, toset):
    return set, iterset, toset


@pytest.fixture
def mset(sets):
    return op2.MixedSet(sets)


@pytest.fixture(params=['sets', 'mset', 'gen'])
def msets(sets, mset, request):
    return {'sets': sets, 'mset': mset, 'gen': iter(sets)}[request.param]


@pytest.fixture(params=[1, 2, (2, 3)])
def dset(request, set):
    return op2.DataSet(set, request.param, 'dfoo')


@pytest.fixture
def diterset(iterset):
    return op2.DataSet(iterset, 1, 'diterset')


@pytest.fixture
def dtoset(toset):
    return op2.DataSet(toset, 1, 'dtoset')


@pytest.fixture
def dsets(dset, diterset, dtoset):
    return dset, diterset, dtoset


@pytest.fixture
def mdset(dsets):
    return op2.MixedDataSet(dsets)


@pytest.fixture
def dat(dtoset):
    return op2.Dat(dtoset, np.arange(dtoset.cdim * dtoset.size, dtype=np.int32))


@pytest.fixture
def dats(dtoset, dset):
    return op2.Dat(dtoset), op2.Dat(dset)


@pytest.fixture
def mdat(dats):
    return op2.MixedDat(dats)


@pytest.fixture
def m_iterset_toset(iterset, toset):
    return op2.Map(iterset, toset, 2, [1] * 2 * iterset.size, 'm_iterset_toset')


@pytest.fixture
def m_iterset_set(iterset, set):
    return op2.Map(iterset, set, 2, [1] * 2 * iterset.size, 'm_iterset_set')


@pytest.fixture
def m_set_toset(set, toset):
    return op2.Map(set, toset, 1, [1] * set.size, 'm_set_toset')


@pytest.fixture
def m_set_set(set):
    return op2.Map(set, set, 1, [1] * set.size, 'm_set_set')


@pytest.fixture
def maps(m_iterset_toset, m_iterset_set):
    return m_iterset_toset, m_iterset_set


@pytest.fixture
def mmap(maps):
    return op2.MixedMap(maps)


@pytest.fixture
def const(request):
    c = op2.Const(1, 1, 'test_const_nonunique_name')
    request.addfinalizer(c.remove_from_namespace)
    return c


@pytest.fixture
def mds(dtoset, set):
    return op2.MixedDataSet((dtoset, set))


# pytest doesn't currently support using fixtures are paramters to tests
# or other fixtures. We have to work around that by requesting fixtures
# by name
@pytest.fixture(params=[('mds', 'mds', 'mmap', 'mmap'),
                        ('mds', 'dtoset', 'mmap', 'm_iterset_toset'),
                        ('dtoset', 'mds', 'm_iterset_toset', 'mmap')])
def ms(request):
    rds, cds, rm, cm = [request.getfuncargvalue(p) for p in request.param]
    return op2.Sparsity((rds, cds), (rm, cm))


@pytest.fixture
def sparsity(m_iterset_toset, dtoset):
    return op2.Sparsity((dtoset, dtoset), (m_iterset_toset, m_iterset_toset))


@pytest.fixture
def mat(sparsity):
    return op2.Mat(sparsity)


@pytest.fixture
def diag_mat(toset):
    return op2.Mat(op2.Sparsity(toset, op2.Map(toset, toset, 1, np.arange(toset.size))))


@pytest.fixture
def mmat(ms):
    return op2.Mat(ms)


@pytest.fixture
def g():
    return op2.Global(1, 1)


class TestClassAPI:

    """Do PyOP2 classes behave like normal classes?"""

    def test_isinstance(self, backend, set, dat):
        "isinstance should behave as expected."
        assert isinstance(set, op2.Set)
        assert isinstance(dat, op2.Dat)
        assert not isinstance(set, op2.Dat)
        assert not isinstance(dat, op2.Set)

    def test_issubclass(self, backend, set, dat):
        "issubclass should behave as expected"
        assert issubclass(type(set), op2.Set)
        assert issubclass(type(dat), op2.Dat)
        assert not issubclass(type(set), op2.Dat)
        assert not issubclass(type(dat), op2.Set)


class TestInitAPI:

    """
    Init API unit tests
    """

    def test_noninit(self):
        "RuntimeError should be raised when using op2 before calling init."
        with pytest.raises(RuntimeError):
            op2.Set(1)

    def test_not_initialised(self):
        "PyOP2 should report not initialised before op2.init has been called."
        assert not op2.initialised()

    def test_invalid_init(self):
        "init should not accept an invalid backend."
        with pytest.raises(ImportError):
            op2.init(backend='invalid_backend')

    def test_init(self, backend):
        "init should correctly set the backend."
        assert op2.backends.get_backend() == 'pyop2.' + backend

    def test_initialised(self, backend):
        "PyOP2 should report initialised after op2.init has been called."
        assert op2.initialised()

    def test_double_init(self, backend):
        "Calling init again with the same backend should update the configuration."
        op2.init(backend=backend, foo='bar')
        assert op2.backends.get_backend() == 'pyop2.' + backend
        assert op2.configuration['foo'] == 'bar'

    def test_change_backend_fails(self, backend):
        "Calling init again with a different backend should fail."
        with pytest.raises(RuntimeError):
            op2.init(backend='other')


class TestMPIAPI:

    """
    Init API unit tests
    """

    def test_running_sequentially(self, backend):
        "MPI.parallel should return false if running sequentially."
        assert not op2.MPI.parallel

    def test_set_mpi_comm_int(self, backend):
        "int should be converted to mpi4py MPI communicator."
        oldcomm = op2.MPI.comm
        op2.MPI.comm = 1
        assert isinstance(op2.MPI.comm, MPI.Comm)
        op2.MPI.comm = oldcomm

    def test_set_mpi_comm_mpi4py(self, backend):
        "Setting an mpi4py MPI communicator should be allowed."
        oldcomm = op2.MPI.comm
        op2.MPI.comm = MPI.COMM_SELF
        assert isinstance(op2.MPI.comm, MPI.Comm)
        op2.MPI.comm = oldcomm

    def test_set_mpi_comm_invalid_type(self, backend):
        "Invalid MPI communicator type should raise TypeError."
        with pytest.raises(TypeError):
            op2.MPI.comm = None


class TestAccessAPI:

    """
    Access API unit tests
    """

    @pytest.mark.parametrize("mode", base.Access._modes)
    def test_access_repr(self, backend, mode):
        "Access repr should produce an Access object when eval'd."
        from pyop2.base import Access
        assert isinstance(eval(repr(Access(mode))), Access)

    @pytest.mark.parametrize("mode", base.Access._modes)
    def test_access_str(self, backend, mode):
        "Access should have the expected string representation."
        assert str(base.Access(mode)) == "OP2 Access: %s" % mode

    def test_illegal_access(self, backend):
        "Illegal access modes should raise an exception."
        with pytest.raises(exceptions.ModeValueError):
            base.Access('ILLEGAL_ACCESS')


class TestArgAPI:

    """
    Arg API unit tests
    """

    def test_arg_split_dat(self, backend, dat, m_iterset_toset):
        arg = dat(op2.READ, m_iterset_toset)
        for a in arg.split:
            assert a == arg

    def test_arg_split_mdat(self, backend, mdat, mmap):
        arg = mdat(op2.READ, mmap)
        for a, d in zip(arg.split, mdat):
            assert a.data == d

    def test_arg_split_mat(self, backend, skip_opencl, mat, m_iterset_toset):
        arg = mat(op2.INC, (m_iterset_toset[0], m_iterset_toset[0]))
        for a in arg.split:
            assert a == arg

    def test_arg_split_global(self, backend, g):
        arg = g(op2.READ)
        for a in arg.split:
            assert a == arg

    def test_arg_eq_dat(self, backend, dat, m_iterset_toset):
        assert dat(op2.READ, m_iterset_toset) == dat(op2.READ, m_iterset_toset)
        assert dat(op2.READ, m_iterset_toset[0]) == dat(op2.READ, m_iterset_toset[0])
        assert not dat(op2.READ, m_iterset_toset) != dat(op2.READ, m_iterset_toset)
        assert not dat(op2.READ, m_iterset_toset[0]) != dat(op2.READ, m_iterset_toset[0])

    def test_arg_ne_dat_idx(self, backend, dat, m_iterset_toset):
        a1 = dat(op2.READ, m_iterset_toset[0])
        a2 = dat(op2.READ, m_iterset_toset[1])
        assert a1 != a2
        assert not a1 == a2

    def test_arg_ne_dat_mode(self, backend, dat, m_iterset_toset):
        a1 = dat(op2.READ, m_iterset_toset)
        a2 = dat(op2.WRITE, m_iterset_toset)
        assert a1 != a2
        assert not a1 == a2

    def test_arg_ne_dat_map(self, backend, dat, m_iterset_toset):
        m2 = op2.Map(m_iterset_toset.iterset, m_iterset_toset.toset, 1,
                     np.ones(m_iterset_toset.iterset.size))
        assert dat(op2.READ, m_iterset_toset) != dat(op2.READ, m2)
        assert not dat(op2.READ, m_iterset_toset) == dat(op2.READ, m2)

    def test_arg_eq_mat(self, backend, skip_opencl, mat, m_iterset_toset):
        a1 = mat(op2.INC, (m_iterset_toset[0], m_iterset_toset[0]))
        a2 = mat(op2.INC, (m_iterset_toset[0], m_iterset_toset[0]))
        assert a1 == a2
        assert not a1 != a2

    def test_arg_ne_mat_idx(self, backend, skip_opencl, mat, m_iterset_toset):
        a1 = mat(op2.INC, (m_iterset_toset[0], m_iterset_toset[0]))
        a2 = mat(op2.INC, (m_iterset_toset[1], m_iterset_toset[1]))
        assert a1 != a2
        assert not a1 == a2

    def test_arg_ne_mat_mode(self, backend, skip_opencl, mat, m_iterset_toset):
        a1 = mat(op2.INC, (m_iterset_toset[0], m_iterset_toset[0]))
        a2 = mat(op2.WRITE, (m_iterset_toset[0], m_iterset_toset[0]))
        assert a1 != a2
        assert not a1 == a2


class TestSetAPI:

    """
    Set API unit tests
    """

    def test_set_illegal_size(self, backend):
        "Set size should be int."
        with pytest.raises(exceptions.SizeTypeError):
            op2.Set('illegalsize')

    def test_set_illegal_name(self, backend):
        "Set name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Set(1, 2)

    def test_set_iter(self, backend, set):
        "Set should be iterable and yield self."
        for s in set:
            assert s is set

    def test_set_len(self, backend, set):
        "Set len should be 1."
        assert len(set) == 1

    def test_set_repr(self, backend, set):
        "Set repr should produce a Set object when eval'd."
        from pyop2.op2 import Set  # noqa: needed by eval
        assert isinstance(eval(repr(set)), op2.Set)

    def test_set_str(self, backend, set):
        "Set should have the expected string representation."
        assert str(set) == "OP2 Set: %s with size %s" % (set.name, set.size)

    def test_set_eq(self, backend, set):
        "The equality test for sets is identity, not attribute equality"
        assert set == set
        assert not set != set

    def test_set_ne(self, backend, set):
        "Sets with the same attributes should not be equal if not identical."
        setcopy = op2.Set(set.size, set.name)
        assert set != setcopy
        assert not set == setcopy

    def test_dset_in_set(self, backend, set, dset):
        "The in operator should indicate compatibility of DataSet and Set"
        assert dset in set

    def test_dset_not_in_set(self, backend, dset):
        "The in operator should indicate incompatibility of DataSet and Set"
        assert dset not in op2.Set(5, 'bar')

    def test_set_exponentiation_builds_dset(self, backend, set):
        "The exponentiation operator should build a DataSet"
        dset = set ** 1
        assert isinstance(dset, op2.DataSet)
        assert dset.cdim == 1

        dset = set ** 3
        assert dset.cdim == 3


class TestExtrudedSetAPI:
    """
    ExtrudedSet API tests
    """
    def test_illegal_layers_arg(self, backend, set):
        """Must pass at least 2 as a layers argument"""
        with pytest.raises(exceptions.SizeTypeError):
            op2.ExtrudedSet(set, 1)

    def test_illegal_set_arg(self, backend):
        """Extuded Set should be build on a Set"""
        with pytest.raises(TypeError):
            op2.ExtrudedSet(1, 3)

    def test_set_compatiblity(self, backend, set, iterset):
        """The set an extruded set was built on should be contained in it"""
        e = op2.ExtrudedSet(set, 5)
        assert set in e
        assert iterset not in e

    def test_iteration_compatibility(self, backend, iterset, m_iterset_toset, m_iterset_set, dats):
        """It should be possible to iterate over an extruded set reading dats
           defined on the base set (indirectly)."""
        e = op2.ExtrudedSet(iterset, 5)
        k = op2.Kernel('void k() { }', 'k')
        dat1, dat2 = dats
        base.ParLoop(k, e, dat1(op2.READ, m_iterset_toset))
        base.ParLoop(k, e, dat2(op2.READ, m_iterset_set))

    def test_iteration_incompatibility(self, backend, set, m_iterset_toset, dat):
        """It should not be possible to iteratve over an extruded set reading
           dats not defined on the base set (indirectly)."""
        e = op2.ExtrudedSet(set, 5)
        k = op2.Kernel('void k() { }', 'k')
        with pytest.raises(exceptions.MapValueError):
            base.ParLoop(k, e, dat(op2.READ, m_iterset_toset))


class TestSubsetAPI:
    """
    Subset API unit tests
    """

    def test_illegal_set_arg(self, backend):
        "The subset constructor checks arguments."
        with pytest.raises(TypeError):
            op2.Subset("fail", [0, 1])

    def test_out_of_bounds_index(self, backend, set):
        "The subset constructor checks indices are correct."
        with pytest.raises(exceptions.SubsetIndexOutOfBounds):
            op2.Subset(set, range(set.total_size + 1))

    def test_invalid_index(self, backend, set):
        "The subset constructor checks indices are correct."
        with pytest.raises(exceptions.SubsetIndexOutOfBounds):
            op2.Subset(set, [-1])

    def test_empty_subset(self, backend, set):
        "Subsets can be empty."
        ss = op2.Subset(set, [])
        assert len(ss.indices) == 0

    def test_index_construction(self, backend, set):
        "We should be able to construct a Subset by indexing a Set."
        ss = set(0, 1)
        ss2 = op2.Subset(set, [0, 1])
        assert_equal(ss.indices, ss2.indices)

        ss = set(0)
        ss2 = op2.Subset(set, [0])
        assert_equal(ss.indices, ss2.indices)

        ss = set(np.arange(5))
        ss2 = op2.Subset(set, np.arange(5))
        assert_equal(ss.indices, ss2.indices)

    def test_indices_duplicate_removed(self, backend, set):
        "The subset constructor voids duplicate indices)"
        ss = op2.Subset(set, [0, 0, 1, 1])
        assert np.sum(ss.indices == 0) == 1
        assert np.sum(ss.indices == 1) == 1

    def test_indices_sorted(self, backend, set):
        "The subset constructor sorts indices)"
        ss = op2.Subset(set, [0, 4, 1, 2, 3])
        assert_equal(ss.indices, range(5))

        ss2 = op2.Subset(set, range(5))
        assert_equal(ss.indices, ss2.indices)


class TestMixedSetAPI:

    """
    MixedSet API unit tests
    """

    def test_mixed_set_illegal_set(self, backend):
        "MixedSet sets should be of type Set."
        with pytest.raises(TypeError):
            op2.MixedSet(('foo', 'bar'))

    def test_mixed_set_getitem(self, backend, sets):
        "MixedSet should return the corresponding Set when indexed."
        mset = op2.MixedSet(sets)
        for i, s in enumerate(sets):
            assert mset[i] == s

    def test_mixed_set_split(self, backend, sets):
        "MixedSet split should return a tuple of the Sets."
        assert op2.MixedSet(sets).split == sets

    def test_mixed_set_core_size(self, backend, mset):
        "MixedSet core_size should return the sum of the Set core_sizes."
        assert mset.core_size == sum(s.core_size for s in mset)

    def test_mixed_set_size(self, backend, mset):
        "MixedSet size should return the sum of the Set sizes."
        assert mset.size == sum(s.size for s in mset)

    def test_mixed_set_exec_size(self, backend, mset):
        "MixedSet exec_size should return the sum of the Set exec_sizes."
        assert mset.exec_size == sum(s.exec_size for s in mset)

    def test_mixed_set_total_size(self, backend, mset):
        "MixedSet total_size should return the sum of the Set total_sizes."
        assert mset.total_size == sum(s.total_size for s in mset)

    def test_mixed_set_sizes(self, backend, mset):
        "MixedSet sizes should return a tuple of the Set sizes."
        assert mset.sizes == (mset.core_size, mset.size, mset.exec_size, mset.total_size)

    def test_mixed_set_name(self, backend, mset):
        "MixedSet name should return a tuple of the Set names."
        assert mset.name == tuple(s.name for s in mset)

    def test_mixed_set_halo(self, backend, mset):
        "MixedSet halo should be None when running sequentially."
        assert mset.halo is None

    def test_mixed_set_layers(self, backend, mset):
        "MixedSet layers should return the layers of the first Set."
        assert mset.layers == mset[0].layers

    def test_mixed_set_layers_must_match(self, backend, sets):
        "All components of a MixedSet must have the same number of layers."
        sets = [op2.ExtrudedSet(s, layers=i+4) for i, s in enumerate(sets)]
        with pytest.raises(AssertionError):
            op2.MixedSet(sets)

    def test_mixed_set_iter(self, backend, mset, sets):
        "MixedSet should be iterable and yield the Sets."
        assert tuple(s for s in mset) == sets

    def test_mixed_set_len(self, backend, sets):
        "MixedSet should have length equal to the number of contained Sets."
        assert len(op2.MixedSet(sets)) == len(sets)

    def test_mixed_set_pow_int(self, backend, mset):
        "MixedSet should implement ** operator returning a MixedDataSet."
        assert mset ** 1 == op2.MixedDataSet([s ** 1 for s in mset])

    def test_mixed_set_pow_seq(self, backend, mset):
        "MixedSet should implement ** operator returning a MixedDataSet."
        assert mset ** ((1,) * len(mset)) == op2.MixedDataSet([s ** 1 for s in mset])

    def test_mixed_set_pow_gen(self, backend, mset):
        "MixedSet should implement ** operator returning a MixedDataSet."
        assert mset ** (1 for _ in mset) == op2.MixedDataSet([s ** 1 for s in mset])

    def test_mixed_set_eq(self, backend, sets):
        "MixedSets created from the same Sets should compare equal."
        assert op2.MixedSet(sets) == op2.MixedSet(sets)
        assert not op2.MixedSet(sets) != op2.MixedSet(sets)

    def test_mixed_set_ne(self, backend, set, iterset, toset):
        "MixedSets created from different Sets should not compare equal."
        assert op2.MixedSet((set, iterset, toset)) != op2.MixedSet((set, toset, iterset))
        assert not op2.MixedSet((set, iterset, toset)) == op2.MixedSet((set, toset, iterset))

    def test_mixed_set_ne_set(self, backend, sets):
        "A MixedSet should not compare equal to a Set."
        assert op2.MixedSet(sets) != sets[0]
        assert not op2.MixedSet(sets) == sets[0]

    def test_mixed_set_repr(self, backend, mset):
        "MixedSet repr should produce a MixedSet object when eval'd."
        from pyop2.op2 import Set, MixedSet  # noqa: needed by eval
        assert isinstance(eval(repr(mset)), base.MixedSet)

    def test_mixed_set_str(self, backend, mset):
        "MixedSet should have the expected string representation."
        assert str(mset) == "OP2 MixedSet composed of Sets: %s" % (mset._sets,)


class TestDataSetAPI:
    """
    DataSet API unit tests
    """

    def test_dset_illegal_dim(self, backend, iterset):
        "DataSet dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.DataSet(iterset, 'illegaldim')

    def test_dset_illegal_dim_tuple(self, backend, iterset):
        "DataSet dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.DataSet(iterset, (1, 'illegaldim'))

    def test_dset_illegal_name(self, backend, iterset):
        "DataSet name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.DataSet(iterset, 1, 2)

    def test_dset_default_dim(self, backend, iterset):
        "DataSet constructor should default dim to (1,)."
        assert op2.DataSet(iterset).dim == (1,)

    def test_dset_dim(self, backend, iterset):
        "DataSet constructor should create a dim tuple."
        s = op2.DataSet(iterset, 1)
        assert s.dim == (1,)

    def test_dset_dim_list(self, backend, iterset):
        "DataSet constructor should create a dim tuple from a list."
        s = op2.DataSet(iterset, [2, 3])
        assert s.dim == (2, 3)

    def test_dset_iter(self, backend, dset):
        "DataSet should be iterable and yield self."
        for s in dset:
            assert s is dset

    def test_dset_len(self, backend, dset):
        "DataSet len should be 1."
        assert len(dset) == 1

    def test_dset_repr(self, backend, dset):
        "DataSet repr should produce a Set object when eval'd."
        from pyop2.op2 import Set, DataSet  # noqa: needed by eval
        assert isinstance(eval(repr(dset)), op2.DataSet)

    def test_dset_str(self, backend, dset):
        "DataSet should have the expected string representation."
        assert str(dset) == "OP2 DataSet: %s on set %s, with dim %s" \
            % (dset.name, dset.set, dset.dim)

    def test_dset_eq(self, backend, dset):
        "The equality test for DataSets is same dim and same set"
        dsetcopy = op2.DataSet(dset.set, dset.dim)
        assert dsetcopy == dset
        assert not dsetcopy != dset

    def test_dset_ne_set(self, backend, dset):
        "DataSets with the same dim but different Sets are not equal."
        dsetcopy = op2.DataSet(op2.Set(dset.set.size), dset.dim)
        assert dsetcopy != dset
        assert not dsetcopy == dset

    def test_dset_ne_dim(self, backend, dset):
        "DataSets with the same Set but different dims are not equal."
        dsetcopy = op2.DataSet(dset.set, tuple(d + 1 for d in dset.dim))
        assert dsetcopy != dset
        assert not dsetcopy == dset

    def test_dat_in_dset(self, backend, dset):
        "The in operator should indicate compatibility of DataSet and Set"
        assert op2.Dat(dset) in dset

    def test_dat_not_in_dset(self, backend, dset):
        "The in operator should indicate incompatibility of DataSet and Set"
        assert op2.Dat(dset) not in op2.DataSet(op2.Set(5, 'bar'))


class TestMixedDataSetAPI:
    """
    MixedDataSet API unit tests
    """

    @pytest.mark.parametrize('arg', ['illegalarg', (set, 'illegalarg'),
                                     iter((set, 'illegalarg'))])
    def test_mixed_dset_illegal_arg(self, backend, arg):
        """Constructing a MixedDataSet from anything other than a MixedSet or
        an iterable of Sets and/or DataSets should fail."""
        with pytest.raises(TypeError):
            op2.MixedDataSet(arg)

    @pytest.mark.parametrize('dims', ['illegaldim', (1, 2, 'illegaldim')])
    def test_mixed_dset_dsets_illegal_dims(self, backend, dsets, dims):
        """When constructing a MixedDataSet from an iterable of DataSets it is
        an error to specify dims."""
        with pytest.raises((TypeError, ValueError)):
            op2.MixedDataSet(dsets, dims)

    def test_mixed_dset_dsets_dims(self, backend, dsets):
        """When constructing a MixedDataSet from an iterable of DataSets it is
        an error to specify dims."""
        with pytest.raises(TypeError):
            op2.MixedDataSet(dsets, 1)

    def test_mixed_dset_upcast_sets(self, backend, msets, mset):
        """Constructing a MixedDataSet from an iterable/iterator of Sets or
        MixedSet should upcast."""
        assert op2.MixedDataSet(msets) == mset ** 1

    def test_mixed_dset_sets_and_dsets(self, backend, set, dset):
        """Constructing a MixedDataSet from an iterable with a mixture of
        Sets and DataSets should upcast the Sets."""
        assert op2.MixedDataSet((set, dset)).split == (set ** 1, dset)

    def test_mixed_dset_sets_and_dsets_gen(self, backend, set, dset):
        """Constructing a MixedDataSet from an iterable with a mixture of
        Sets and DataSets should upcast the Sets."""
        assert op2.MixedDataSet(iter((set, dset))).split == (set ** 1, dset)

    def test_mixed_dset_dims_default_to_one(self, backend, msets, mset):
        """Constructing a MixedDataSet from an interable/iterator of Sets or
        MixedSet without dims should default them to 1."""
        assert op2.MixedDataSet(msets).dim == ((1,),) * len(mset)

    def test_mixed_dset_dims_int(self, backend, msets, mset):
        """Construct a MixedDataSet from an iterator/iterable of Sets and a
        MixedSet with dims as an int."""
        assert op2.MixedDataSet(msets, 2).dim == ((2,),) * len(mset)

    def test_mixed_dset_dims_gen(self, backend, msets, mset):
        """Construct a MixedDataSet from an iterator/iterable of Sets and a
        MixedSet with dims as a generator."""
        dims = (2 for _ in mset)
        assert op2.MixedDataSet(msets, dims).dim == ((2,),) * len(mset)

    def test_mixed_dset_dims_iterable(self, backend, msets):
        """Construct a MixedDataSet from an iterator/iterable of Sets and a
        MixedSet with dims as an iterable."""
        dims = ((2,), (2, 2), (1,))
        assert op2.MixedDataSet(msets, dims).dim == dims

    def test_mixed_dset_dims_mismatch(self, backend, msets, sets):
        """Constructing a MixedDataSet from an iterable/iterator of Sets and a
        MixedSet with mismatching number of dims should raise ValueError."""
        with pytest.raises(ValueError):
            op2.MixedDataSet(msets, range(1, len(sets)))

    def test_mixed_dset_getitem(self, backend, mdset):
        "MixedDataSet should return the corresponding DataSet when indexed."
        for i, ds in enumerate(mdset):
            assert mdset[i] == ds

    def test_mixed_dset_split(self, backend, dsets):
        "MixedDataSet split should return a tuple of the DataSets."
        assert op2.MixedDataSet(dsets).split == dsets

    def test_mixed_dset_dim(self, backend, mdset):
        "MixedDataSet dim should return a tuple of the DataSet dims."
        assert mdset.dim == tuple(s.dim for s in mdset)

    def test_mixed_dset_cdim(self, backend, mdset):
        "MixedDataSet cdim should return the sum of the DataSet cdims."
        assert mdset.cdim == sum(s.cdim for s in mdset)

    def test_mixed_dset_name(self, backend, mdset):
        "MixedDataSet name should return a tuple of the DataSet names."
        assert mdset.name == tuple(s.name for s in mdset)

    def test_mixed_dset_set(self, backend, mset):
        "MixedDataSet set should return a MixedSet."
        assert op2.MixedDataSet(mset).set == mset

    def test_mixed_dset_iter(self, backend, mdset, dsets):
        "MixedDataSet should be iterable and yield the DataSets."
        assert tuple(s for s in mdset) == dsets

    def test_mixed_dset_len(self, backend, dsets):
        """MixedDataSet should have length equal to the number of contained
        DataSets."""
        assert len(op2.MixedDataSet(dsets)) == len(dsets)

    def test_mixed_dset_eq(self, backend, dsets):
        "MixedDataSets created from the same DataSets should compare equal."
        assert op2.MixedDataSet(dsets) == op2.MixedDataSet(dsets)
        assert not op2.MixedDataSet(dsets) != op2.MixedDataSet(dsets)

    def test_mixed_dset_ne(self, backend, dset, diterset, dtoset):
        "MixedDataSets created from different DataSets should not compare equal."
        mds1 = op2.MixedDataSet((dset, diterset, dtoset))
        mds2 = op2.MixedDataSet((dset, dtoset, diterset))
        assert mds1 != mds2
        assert not mds1 == mds2

    def test_mixed_dset_ne_dset(self, backend, diterset, dtoset):
        "MixedDataSets should not compare equal to a scalar DataSet."
        assert op2.MixedDataSet((diterset, dtoset)) != diterset
        assert not op2.MixedDataSet((diterset, dtoset)) == diterset

    def test_mixed_dset_repr(self, backend, mdset):
        "MixedDataSet repr should produce a MixedDataSet object when eval'd."
        from pyop2.op2 import Set, DataSet, MixedDataSet  # noqa: needed by eval
        assert isinstance(eval(repr(mdset)), base.MixedDataSet)

    def test_mixed_dset_str(self, backend, mdset):
        "MixedDataSet should have the expected string representation."
        assert str(mdset) == "OP2 MixedDataSet composed of DataSets: %s" % (mdset._dsets,)


class TestDatAPI:

    """
    Dat API unit tests
    """

    def test_dat_illegal_set(self, backend):
        "Dat set should be DataSet."
        with pytest.raises(exceptions.DataSetTypeError):
            op2.Dat('illegalset', 1)

    def test_dat_illegal_name(self, backend, dset):
        "Dat name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Dat(dset, name=2)

    def test_dat_initialise_data(self, backend, dset):
        """Dat initilialised without the data should initialise data with the
        correct size and type."""
        d = op2.Dat(dset)
        assert d.data.size == dset.size * dset.cdim and d.data.dtype == np.float64

    def test_dat_initialise_data_type(self, backend, dset):
        """Dat intiialised without the data but with specified type should
        initialise its data with the correct type."""
        d = op2.Dat(dset, dtype=np.int32)
        assert d.data.dtype == np.int32

    @pytest.mark.parametrize("mode", [op2.MAX, op2.MIN])
    def test_dat_arg_illegal_mode(self, backend, dat, mode):
        """Dat __call__ should not allow access modes not allowed for a Dat."""
        with pytest.raises(exceptions.ModeValueError):
            dat(mode)

    def test_dat_subscript(self, backend, dat):
        """Extracting component 0 of a Dat should yield self."""
        assert dat[0] is dat

    def test_dat_illegal_subscript(self, backend, dat):
        """Extracting component 0 of a Dat should yield self."""
        with pytest.raises(exceptions.IndexValueError):
            dat[1]

    def test_dat_arg_default_map(self, backend, dat):
        """Dat __call__ should default the Arg map to None if not given."""
        assert dat(op2.READ).map is None

    def test_dat_arg_illegal_map(self, backend, dset):
        """Dat __call__ should not allow a map with a toset other than this
        Dat's set."""
        d = op2.Dat(dset)
        set1 = op2.Set(3)
        set2 = op2.Set(2)
        to_set2 = op2.Map(set1, set2, 1, [0, 0, 0])
        with pytest.raises(exceptions.MapValueError):
            d(op2.READ, to_set2)

    def test_dat_on_set_builds_dim_one_dataset(self, backend, set):
        """If a Set is passed as the dataset argument, it should be
        converted into a Dataset with dim=1"""
        d = op2.Dat(set)
        assert d.cdim == 1
        assert isinstance(d.dataset, op2.DataSet)
        assert d.dataset.cdim == 1

    def test_dat_dtype_type(self, backend, dset):
        "The type of a Dat's dtype property should by numpy.dtype."
        d = op2.Dat(dset)
        assert type(d.dtype) == np.dtype
        d = op2.Dat(dset, [1.0] * dset.size * dset.cdim)
        assert type(d.dtype) == np.dtype

    def test_dat_split(self, backend, dat):
        "Splitting a Dat should yield a tuple with self"
        for d in dat.split:
            d == dat

    def test_dat_dtype(self, backend, dset):
        "Default data type should be numpy.float64."
        d = op2.Dat(dset)
        assert d.dtype == np.double

    def test_dat_float(self, backend, dset):
        "Data type for float data should be numpy.float64."
        d = op2.Dat(dset, [1.0] * dset.size * dset.cdim)
        assert d.dtype == np.double

    def test_dat_int(self, backend, dset):
        "Data type for int data should be numpy.int."
        d = op2.Dat(dset, [1] * dset.size * dset.cdim)
        assert d.dtype == np.int

    def test_dat_convert_int_float(self, backend, dset):
        "Explicit float type should override NumPy's default choice of int."
        d = op2.Dat(dset, [1] * dset.size * dset.cdim, np.double)
        assert d.dtype == np.float64

    def test_dat_convert_float_int(self, backend, dset):
        "Explicit int type should override NumPy's default choice of float."
        d = op2.Dat(dset, [1.5] * dset.size * dset.cdim, np.int32)
        assert d.dtype == np.int32

    def test_dat_illegal_dtype(self, backend, dset):
        "Illegal data type should raise DataTypeError."
        with pytest.raises(exceptions.DataTypeError):
            op2.Dat(dset, dtype='illegal_type')

    def test_dat_illegal_length(self, backend, dset):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Dat(dset, [1] * (dset.size * dset.cdim + 1))

    def test_dat_reshape(self, backend, dset):
        "Data should be reshaped according to the set's dim."
        d = op2.Dat(dset, [1.0] * dset.size * dset.cdim)
        shape = (dset.size,) + (() if dset.cdim == 1 else dset.dim)
        assert d.data.shape == shape

    def test_dat_properties(self, backend, dset):
        "Dat constructor should correctly set attributes."
        d = op2.Dat(dset, [1] * dset.size * dset.cdim, 'double', 'bar')
        assert d.dataset.set == dset.set and d.dtype == np.float64 and \
            d.name == 'bar' and d.data.sum() == dset.size * dset.cdim

    def test_dat_eq(self, backend, dset):
        """Dats should compare equal if defined on the same DataSets and
        having the same data."""
        assert op2.Dat(dset) == op2.Dat(dset)
        assert not op2.Dat(dset) != op2.Dat(dset)

    def test_dat_ne_dset(self, backend):
        """Dats should not compare equal if defined on different DataSets."""
        assert op2.Dat(op2.Set(3)) != op2.Dat(op2.Set(3))
        assert not op2.Dat(op2.Set(3)) == op2.Dat(op2.Set(3))

    def test_dat_ne_dtype(self, backend, dset):
        """Dats should not compare equal when having data of different
        dtype."""
        assert op2.Dat(dset, dtype=np.int64) != op2.Dat(dset, dtype=np.float64)
        assert not op2.Dat(dset, dtype=np.int64) == op2.Dat(dset, dtype=np.float64)

    def test_dat_ne_data(self, backend, dset):
        """Dats should not compare equal when having different data."""
        d1, d2 = op2.Dat(dset), op2.Dat(dset)
        d1.data[0] = -1.0
        assert d1 != d2
        assert not d1 == d2

    def test_dat_iter(self, backend, dat):
        "Dat should be iterable and yield self."
        for d in dat:
            assert d is dat

    def test_dat_len(self, backend, dat):
        "Dat len should be 1."
        assert len(dat) == 1

    def test_dat_repr(self, backend, dat):
        "Dat repr should produce a Dat object when eval'd."
        from pyop2.op2 import Dat, DataSet, Set  # noqa: needed by eval
        from numpy import dtype  # noqa: needed by eval
        assert isinstance(eval(repr(dat)), op2.Dat)

    def test_dat_str(self, backend, dset):
        "Dat should have the expected string representation."
        d = op2.Dat(dset, dtype='double', name='bar')
        s = "OP2 Dat: %s on (%s) with datatype %s" \
            % (d.name, d.dataset, d.data.dtype.name)
        assert str(d) == s

    def test_dat_ro_accessor(self, backend, dat):
        "Attempting to set values through the RO accessor should raise an error."
        x = dat.data_ro
        with pytest.raises((RuntimeError, ValueError)):
            x[0] = 1

    def test_dat_ro_write_accessor(self, backend, dat):
        "Re-accessing the data in writeable form should be allowed."
        x = dat.data_ro
        with pytest.raises((RuntimeError, ValueError)):
            x[0] = 1
        x = dat.data
        x[0] = -100
        assert (dat.data_ro[0] == -100).all()

    def test_dat_lazy_allocation(self, backend, dset):
        "Temporary Dats should not allocate storage until accessed."
        d = op2.Dat(dset)
        assert not d._is_allocated

    def test_dat_zero_cdim(self, backend, set):
        "A Dat built on a DataSet with zero dim should be allowed."
        dset = set**0
        d = op2.Dat(dset)
        assert d.shape == (set.total_size, 0)
        assert d.data.size == 0
        assert d.data.shape == (set.total_size, 0)


class TestMixedDatAPI:

    """
    MixedDat API unit tests
    """

    def test_mixed_dat_illegal_arg(self, backend):
        """Constructing a MixedDat from anything other than a MixedSet, a
        MixedDataSet or an iterable of Dats should fail."""
        with pytest.raises(exceptions.DataSetTypeError):
            op2.MixedDat('illegalarg')

    def test_mixed_dat_illegal_dtype(self, backend, set):
        """Constructing a MixedDat from Dats of different dtype should fail."""
        with pytest.raises(exceptions.DataValueError):
            op2.MixedDat((op2.Dat(set, dtype=np.int), op2.Dat(set)))

    def test_mixed_dat_dats(self, backend, dats):
        """Constructing a MixedDat from an iterable of Dats should leave them
        unchanged."""
        assert op2.MixedDat(dats).split == dats

    def test_mixed_dat_dsets(self, backend, mdset):
        """Constructing a MixedDat from an iterable of DataSets should leave
        them unchanged."""
        assert op2.MixedDat(mdset).dataset == mdset

    def test_mixed_dat_upcast_sets(self, backend, mset):
        "Constructing a MixedDat from an iterable of Sets should upcast."
        assert op2.MixedDat(mset).dataset == op2.MixedDataSet(mset)

    def test_mixed_dat_sets_dsets_dats(self, backend, set, dset):
        """Constructing a MixedDat from an iterable of Sets, DataSets and
        Dats should upcast as necessary."""
        dat = op2.Dat(op2.Set(3) ** 2)
        assert op2.MixedDat((set, dset, dat)).split == (op2.Dat(set), op2.Dat(dset), dat)

    def test_mixed_dat_getitem(self, backend, mdat):
        "MixedDat should return the corresponding Dat when indexed."
        for i, d in enumerate(mdat):
            assert mdat[i] == d
        assert mdat[:-1] == tuple(mdat)[:-1]

    def test_mixed_dat_dim(self, backend, mdset):
        "MixedDat dim should return a tuple of the DataSet dims."
        assert op2.MixedDat(mdset).dim == mdset.dim

    def test_mixed_dat_cdim(self, backend, mdset):
        "MixedDat cdim should return a tuple of the DataSet cdims."
        assert op2.MixedDat(mdset).cdim == mdset.cdim

    def test_mixed_dat_soa(self, backend, mdat):
        "MixedDat soa should return a tuple of the Dat soa flags."
        assert mdat.soa == tuple(d.soa for d in mdat)

    def test_mixed_dat_data(self, backend, mdat):
        "MixedDat data should return a tuple of the Dat data arrays."
        assert all((d1 == d2.data).all() for d1, d2 in zip(mdat.data, mdat))

    def test_mixed_dat_data_ro(self, backend, mdat):
        "MixedDat data_ro should return a tuple of the Dat data_ro arrays."
        assert all((d1 == d2.data_ro).all() for d1, d2 in zip(mdat.data_ro, mdat))

    def test_mixed_dat_data_with_halos(self, backend, mdat):
        """MixedDat data_with_halos should return a tuple of the Dat
        data_with_halos arrays."""
        assert all((d1 == d2.data_with_halos).all() for d1, d2 in zip(mdat.data_with_halos, mdat))

    def test_mixed_dat_data_ro_with_halos(self, backend, mdat):
        """MixedDat data_ro_with_halos should return a tuple of the Dat
        data_ro_with_halos arrays."""
        assert all((d1 == d2.data_ro_with_halos).all() for d1, d2 in zip(mdat.data_ro_with_halos, mdat))

    def test_mixed_dat_needs_halo_update(self, backend, mdat):
        """MixedDat needs_halo_update should indicate if at least one contained
        Dat needs a halo update."""
        assert not mdat.needs_halo_update
        mdat[0].needs_halo_update = True
        assert mdat.needs_halo_update

    def test_mixed_dat_needs_halo_update_setter(self, backend, mdat):
        """Setting MixedDat needs_halo_update should set the property for all
        contained Dats."""
        assert not mdat.needs_halo_update
        mdat.needs_halo_update = True
        assert all(d.needs_halo_update for d in mdat)

    def test_mixed_dat_iter(self, backend, mdat, dats):
        "MixedDat should be iterable and yield the Dats."
        assert tuple(s for s in mdat) == dats

    def test_mixed_dat_len(self, backend, dats):
        """MixedDat should have length equal to the number of contained Dats."""
        assert len(op2.MixedDat(dats)) == len(dats)

    def test_mixed_dat_eq(self, backend, dats):
        "MixedDats created from the same Dats should compare equal."
        assert op2.MixedDat(dats) == op2.MixedDat(dats)
        assert not op2.MixedDat(dats) != op2.MixedDat(dats)

    def test_mixed_dat_ne(self, backend, dats):
        "MixedDats created from different Dats should not compare equal."
        mdat1 = op2.MixedDat(dats)
        mdat2 = op2.MixedDat(reversed(dats))
        assert mdat1 != mdat2
        assert not mdat1 == mdat2

    def test_mixed_dat_ne_dat(self, backend, dats):
        "A MixedDat should not compare equal to a Dat."
        assert op2.MixedDat(dats) != dats[0]
        assert not op2.MixedDat(dats) == dats[0]

    def test_mixed_dat_repr(self, backend, mdat):
        "MixedDat repr should produce a MixedDat object when eval'd."
        from pyop2.op2 import Set, DataSet, MixedDataSet, Dat, MixedDat  # noqa: needed by eval
        from numpy import dtype  # noqa: needed by eval
        assert isinstance(eval(repr(mdat)), base.MixedDat)

    def test_mixed_dat_str(self, backend, mdat):
        "MixedDat should have the expected string representation."
        assert str(mdat) == "OP2 MixedDat composed of Dats: %s" % (mdat.split,)


class TestSparsityAPI:

    """
    Sparsity API unit tests
    """

    @pytest.fixture
    def mi(cls, toset):
        iterset = op2.Set(3, 'iterset2')
        return op2.Map(iterset, toset, 1, [1] * iterset.size, 'mi')

    @pytest.fixture
    def dataset2(cls):
        return op2.Set(1, 'dataset2')

    @pytest.fixture
    def md(cls, iterset, dataset2):
        return op2.Map(iterset, dataset2, 1, [0] * iterset.size, 'md')

    @pytest.fixture
    def di(cls, toset):
        return op2.DataSet(toset, 1, 'di')

    @pytest.fixture
    def dd(cls, dataset2):
        return op2.DataSet(dataset2, 1, 'dd')

    @pytest.fixture
    def s(cls, di, mi):
        return op2.Sparsity(di, mi)

    @pytest.fixture
    def mixed_row_sparsity(cls, dtoset, mds, m_iterset_toset, mmap):
        return op2.Sparsity((mds, dtoset), (mmap, m_iterset_toset))

    @pytest.fixture
    def mixed_col_sparsity(cls, dtoset, mds, m_iterset_toset, mmap):
        return op2.Sparsity((dtoset, mds), (m_iterset_toset, mmap))

    def test_sparsity_illegal_rdset(self, backend, di, mi):
        "Sparsity rdset should be a DataSet"
        with pytest.raises(TypeError):
            op2.Sparsity(('illegalrmap', di), (mi, mi))

    def test_sparsity_illegal_cdset(self, backend, di, mi):
        "Sparsity cdset should be a DataSet"
        with pytest.raises(TypeError):
            op2.Sparsity((di, 'illegalrmap'), (mi, mi))

    def test_sparsity_illegal_rmap(self, backend, di, mi):
        "Sparsity rmap should be a Map"
        with pytest.raises(TypeError):
            op2.Sparsity((di, di), ('illegalrmap', mi))

    def test_sparsity_illegal_cmap(self, backend, di, mi):
        "Sparsity cmap should be a Map"
        with pytest.raises(TypeError):
            op2.Sparsity((di, di), (mi, 'illegalcmap'))

    def test_sparsity_illegal_name(self, backend, di, mi):
        "Sparsity name should be a string."
        with pytest.raises(TypeError):
            op2.Sparsity(di, mi, 0)

    def test_sparsity_single_dset(self, backend, di, mi):
        "Sparsity constructor should accept single Map and turn it into tuple"
        s = op2.Sparsity(di, mi, "foo")
        assert (s.maps[0] == (mi, mi) and s.dims[0][0] == (1, 1) and
                s.name == "foo" and s.dsets == (di, di))

    def test_sparsity_set_not_dset(self, backend, di, mi):
        "If we pass a Set, not a DataSet, it default to dimension 1."
        s = op2.Sparsity(mi.toset, mi)
        assert s.maps[0] == (mi, mi) and s.dims[0][0] == (1, 1) \
            and s.dsets == (di, di)

    def test_sparsity_map_pair(self, backend, di, mi):
        "Sparsity constructor should accept a pair of maps"
        s = op2.Sparsity((di, di), (mi, mi), "foo")
        assert (s.maps[0] == (mi, mi) and s.dims[0][0] == (1, 1) and
                s.name == "foo" and s.dsets == (di, di))

    def test_sparsity_map_pair_different_dataset(self, backend, mi, md, di, dd, m_iterset_toset):
        """Sparsity can be built from different row and column maps as long as
        the tosets match the row and column DataSet."""
        s = op2.Sparsity((di, dd), (m_iterset_toset, md), "foo")
        assert (s.maps[0] == (m_iterset_toset, md) and s.dims[0][0] == (1, 1) and
                s.name == "foo" and s.dsets == (di, dd))

    def test_sparsity_unique_map_pairs(self, backend, mi, di):
        "Sparsity constructor should filter duplicate tuples of pairs of maps."
        s = op2.Sparsity((di, di), ((mi, mi), (mi, mi)), "foo")
        assert s.maps == [(mi, mi)] and s.dims[0][0] == (1, 1)

    def test_sparsity_map_pairs_different_itset(self, backend, mi, di, dd, m_iterset_toset):
        "Sparsity constructor should accept maps with different iteration sets"
        maps = ((m_iterset_toset, m_iterset_toset), (mi, mi))
        s = op2.Sparsity((di, di), maps, "foo")
        assert s.maps == list(sorted(maps)) and s.dims[0][0] == (1, 1)

    def test_sparsity_map_pairs_sorted(self, backend, mi, di, dd, m_iterset_toset):
        "Sparsity maps should have a deterministic order."
        s1 = op2.Sparsity((di, di), [(m_iterset_toset, m_iterset_toset), (mi, mi)])
        s2 = op2.Sparsity((di, di), [(mi, mi), (m_iterset_toset, m_iterset_toset)])
        assert s1.maps == s2.maps

    def test_sparsity_illegal_itersets(self, backend, mi, md, di, dd):
        "Both maps in a (rmap,cmap) tuple must have same iteration set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((dd, di), (md, mi))

    def test_sparsity_illegal_row_datasets(self, backend, mi, md, di):
        "All row maps must share the same data set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((di, di), ((mi, mi), (md, mi)))

    def test_sparsity_illegal_col_datasets(self, backend, mi, md, di, dd):
        "All column maps must share the same data set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((di, di), ((mi, mi), (mi, md)))

    def test_sparsity_shape(self, backend, s):
        "Sparsity shape of a single block should be (1, 1)."
        assert s.shape == (1, 1)

    def test_sparsity_iter(self, backend, s):
        "Iterating over a Sparsity of a single block should yield self."
        for bs in s:
            assert bs == s

    def test_sparsity_getitem(self, backend, s):
        "Block 0, 0 of a Sparsity of a single block should be self."
        assert s[0, 0] == s

    def test_sparsity_mmap_iter(self, backend, ms):
        "Iterating a Sparsity should yield the block by row."
        cols = ms.shape[1]
        for i, block in enumerate(ms):
            assert block == ms[i / cols, i % cols]

    def test_sparsity_mmap_getitem(self, backend, ms):
        """Sparsity block i, j should be defined on the corresponding row and
        column DataSets and Maps."""
        for i, (rds, rm) in enumerate(zip(ms.dsets[0], ms.rmaps)):
            for j, (cds, cm) in enumerate(zip(ms.dsets[1], ms.cmaps)):
                block = ms[i, j]
                # Indexing with a tuple and double index is equivalent
                assert block == ms[i][j]
                assert (block.dsets == (rds, cds) and
                        block.maps == [(rm.split[i], cm.split[j])])

    def test_sparsity_mmap_getrow(self, backend, ms):
        """Indexing a Sparsity with a single index should yield a row of
        blocks."""
        for i, (rds, rm) in enumerate(zip(ms.dsets[0], ms.rmaps)):
            for j, (s, cds, cm) in enumerate(zip(ms[i], ms.dsets[1], ms.cmaps)):
                assert (s.dsets == (rds, cds) and
                        s.maps == [(rm.split[i], cm.split[j])])

    def test_sparsity_mmap_shape(self, backend, ms):
        "Sparsity shape of should be the sizes of the mixed space."
        assert ms.shape == (len(ms.dsets[0]), len(ms.dsets[1]))

    def test_sparsity_mmap_illegal_itersets(self, backend, m_iterset_toset,
                                            m_iterset_set, m_set_toset,
                                            m_set_set, mds):
        "Both maps in a (rmap,cmap) tuple must have same iteration set."
        with pytest.raises(RuntimeError):
            op2.Sparsity((mds, mds), (op2.MixedMap((m_iterset_toset, m_iterset_set)),
                                      op2.MixedMap((m_set_toset, m_set_set))))

    def test_sparsity_mmap_illegal_row_datasets(self, backend, m_iterset_toset,
                                                m_iterset_set, m_set_toset, mds):
        "All row maps must share the same data set."
        with pytest.raises(RuntimeError):
            op2.Sparsity((mds, mds), (op2.MixedMap((m_iterset_toset, m_iterset_set)),
                                      op2.MixedMap((m_set_toset, m_set_toset))))

    def test_sparsity_mmap_illegal_col_datasets(self, backend, m_iterset_toset,
                                                m_iterset_set, m_set_toset, mds):
        "All column maps must share the same data set."
        with pytest.raises(RuntimeError):
            op2.Sparsity((mds, mds), (op2.MixedMap((m_set_toset, m_set_toset)),
                                      op2.MixedMap((m_iterset_toset, m_iterset_set))))

    def test_sparsity_repr(self, backend, sparsity):
        "Sparsity should have the expected repr."

        # Note: We can't actually reproduce a Sparsity from its repr because
        # the Sparsity constructor checks that the maps are populated
        r = "Sparsity(%r, %r, %r)" % (sparsity.dsets, sparsity.maps, sparsity.name)
        assert repr(sparsity) == r

    def test_sparsity_str(self, backend, sparsity):
        "Sparsity should have the expected string representation."
        s = "OP2 Sparsity: dsets %s, rmaps %s, cmaps %s, name %s" % \
            (sparsity.dsets, sparsity.rmaps, sparsity.cmaps, sparsity.name)
        assert str(sparsity) == s


class TestMatAPI:

    """
    Mat API unit tests
    """

    skip_backends = ["opencl"]

    def test_mat_illegal_sets(self, backend):
        "Mat sparsity should be a Sparsity."
        with pytest.raises(TypeError):
            op2.Mat('illegalsparsity')

    def test_mat_illegal_name(self, backend, sparsity):
        "Mat name should be string."
        with pytest.raises(sequential.NameTypeError):
            op2.Mat(sparsity, name=2)

    def test_mat_dtype(self, backend, mat):
        "Default data type should be numpy.float64."
        assert mat.dtype == np.double

    def test_mat_properties(self, backend, sparsity):
        "Mat constructor should correctly set attributes."
        m = op2.Mat(sparsity, 'double', 'bar')
        assert m.sparsity == sparsity and  \
            m.dtype == np.float64 and m.name == 'bar'

    def test_mat_mixed(self, backend, mmat, skip_cuda):
        "Default data type should be numpy.float64."
        assert mmat.dtype == np.double

    def test_mat_illegal_maps(self, backend, mat):
        "Mat arg constructor should reject invalid maps."
        wrongmap = op2.Map(op2.Set(2), op2.Set(3), 2, [0, 0, 0, 0])
        with pytest.raises(exceptions.MapValueError):
            mat(op2.INC, (wrongmap[0], wrongmap[1]))

    def test_mat_arg_nonindexed_maps(self, backend, mat, m_iterset_toset):
        "Mat arg constructor should reject nonindexed maps."
        with pytest.raises(TypeError):
            mat(op2.INC, (m_iterset_toset, m_iterset_toset))

    @pytest.mark.parametrize("mode", [op2.READ, op2.RW, op2.MIN, op2.MAX])
    def test_mat_arg_illegal_mode(self, backend, mat, mode, m_iterset_toset):
        """Mat arg constructor should reject illegal access modes."""
        with pytest.raises(exceptions.ModeValueError):
            mat(mode, (m_iterset_toset[op2.i[0]], m_iterset_toset[op2.i[1]]))

    def test_mat_iter(self, backend, mat):
        "Mat should be iterable and yield self."
        for m in mat:
            assert m is mat

    def test_mat_repr(self, backend, mat):
        "Mat should have the expected repr."

        # Note: We can't actually reproduce a Sparsity from its repr because
        # the Sparsity constructor checks that the maps are populated
        r = "Mat(%r, %r, %r)" % (mat.sparsity, mat.dtype, mat.name)
        assert repr(mat) == r

    def test_mat_str(self, backend, mat):
        "Mat should have the expected string representation."
        s = "OP2 Mat: %s, sparsity (%s), datatype %s" \
            % (mat.name, mat.sparsity, mat.dtype.name)
        assert str(mat) == s


class TestConstAPI:

    """
    Const API unit tests
    """

    def test_const_illegal_dim(self, backend):
        "Const dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Const('illegaldim', 1, 'test_const_illegal_dim')

    def test_const_illegal_dim_tuple(self, backend):
        "Const dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Const((1, 'illegaldim'), 1, 'test_const_illegal_dim_tuple')

    def test_const_nonunique_name(self, backend, const):
        "Const names should be unique."
        with pytest.raises(op2.Const.NonUniqueNameError):
            op2.Const(1, 1, 'test_const_nonunique_name')

    def test_const_remove_from_namespace(self, backend):
        "remove_from_namespace should free a global name."
        c = op2.Const(1, 1, 'test_const_remove_from_namespace')
        c.remove_from_namespace()
        c = op2.Const(1, 1, 'test_const_remove_from_namespace')
        c.remove_from_namespace()
        assert c.name == 'test_const_remove_from_namespace'

    def test_const_illegal_name(self, backend):
        "Const name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Const(1, 1, 2)

    def test_const_dim(self, backend):
        "Const constructor should create a dim tuple."
        c = op2.Const(1, 1, 'test_const_dim')
        c.remove_from_namespace()
        assert c.dim == (1,)

    def test_const_dim_list(self, backend):
        "Const constructor should create a dim tuple from a list."
        c = op2.Const([2, 3], [1] * 6, 'test_const_dim_list')
        c.remove_from_namespace()
        assert c.dim == (2, 3)

    def test_const_float(self, backend):
        "Data type for float data should be numpy.float64."
        c = op2.Const(1, 1.0, 'test_const_float')
        c.remove_from_namespace()
        assert c.dtype == np.double

    def test_const_int(self, backend):
        "Data type for int data should be numpy.int."
        c = op2.Const(1, 1, 'test_const_int')
        c.remove_from_namespace()
        assert c.dtype == np.int

    def test_const_convert_int_float(self, backend):
        "Explicit float type should override NumPy's default choice of int."
        c = op2.Const(1, 1, 'test_const_convert_int_float', 'double')
        c.remove_from_namespace()
        assert c.dtype == np.float64

    def test_const_convert_float_int(self, backend):
        "Explicit int type should override NumPy's default choice of float."
        c = op2.Const(1, 1.5, 'test_const_convert_float_int', 'int')
        c.remove_from_namespace()
        assert c.dtype == np.int

    def test_const_illegal_dtype(self, backend):
        "Illegal data type should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Const(1, 'illegal_type', 'test_const_illegal_dtype', 'double')

    @pytest.mark.parametrize("dim", [1, (2, 2)])
    def test_const_illegal_length(self, backend, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Const(
                dim, [1] * (np.prod(dim) + 1), 'test_const_illegal_length_%r' % np.prod(dim))

    def test_const_reshape(self, backend):
        "Data should be reshaped according to dim."
        c = op2.Const((2, 2), [1.0] * 4, 'test_const_reshape')
        c.remove_from_namespace()
        assert c.dim == (2, 2) and c.data.shape == (2, 2)

    def test_const_properties(self, backend):
        "Data constructor should correctly set attributes."
        c = op2.Const((2, 2), [1] * 4, 'baz', 'double')
        c.remove_from_namespace()
        assert c.dim == (2, 2) and c.dtype == np.float64 and c.name == 'baz' \
            and c.data.sum() == 4

    def test_const_setter(self, backend):
        "Setter attribute on data should correct set data value."
        c = op2.Const(1, 1, 'c')
        c.remove_from_namespace()
        c.data = 2
        assert c.data.sum() == 2

    def test_const_setter_malformed_data(self, backend):
        "Setter attribute should reject malformed data."
        c = op2.Const(1, 1, 'c')
        c.remove_from_namespace()
        with pytest.raises(exceptions.DataValueError):
            c.data = [1, 2]

    def test_const_iter(self, backend, const):
        "Const should be iterable and yield self."
        for c in const:
            assert c is const

    def test_const_len(self, backend, const):
        "Const len should be 1."
        assert len(const) == 1

    def test_const_repr(self, backend, const):
        "Const repr should produce a Const object when eval'd."
        from pyop2.op2 import Const  # noqa: needed by eval
        from numpy import array  # noqa: needed by eval
        const.remove_from_namespace()
        c = eval(repr(const))
        assert isinstance(c, op2.Const)
        c.remove_from_namespace()

    def test_const_str(self, backend, const):
        "Const should have the expected string representation."
        s = "OP2 Const: %s of dim %s and type %s with value %s" \
            % (const.name, const.dim, const.data.dtype.name, const.data)
        assert str(const) == s


class TestGlobalAPI:

    """
    Global API unit tests
    """

    def test_global_illegal_dim(self, backend):
        "Global dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Global('illegaldim')

    def test_global_illegal_dim_tuple(self, backend):
        "Global dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Global((1, 'illegaldim'))

    def test_global_illegal_name(self, backend):
        "Global name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Global(1, 1, name=2)

    def test_global_dim(self, backend):
        "Global constructor should create a dim tuple."
        g = op2.Global(1, 1)
        assert g.dim == (1,)

    def test_global_dim_list(self, backend):
        "Global constructor should create a dim tuple from a list."
        g = op2.Global([2, 3], [1] * 6)
        assert g.dim == (2, 3)

    def test_global_float(self, backend):
        "Data type for float data should be numpy.float64."
        g = op2.Global(1, 1.0)
        assert g.dtype == np.double

    def test_global_int(self, backend):
        "Data type for int data should be numpy.int."
        g = op2.Global(1, 1)
        assert g.dtype == np.int

    def test_global_convert_int_float(self, backend):
        "Explicit float type should override NumPy's default choice of int."
        g = op2.Global(1, 1, 'double')
        assert g.dtype == np.float64

    def test_global_convert_float_int(self, backend):
        "Explicit int type should override NumPy's default choice of float."
        g = op2.Global(1, 1.5, 'int')
        assert g.dtype == np.int

    def test_global_illegal_dtype(self, backend):
        "Illegal data type should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Global(1, 'illegal_type', 'double')

    @pytest.mark.parametrize("dim", [1, (2, 2)])
    def test_global_illegal_length(self, backend, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Global(dim, [1] * (np.prod(dim) + 1))

    def test_global_reshape(self, backend):
        "Data should be reshaped according to dim."
        g = op2.Global((2, 2), [1.0] * 4)
        assert g.dim == (2, 2) and g.data.shape == (2, 2)

    def test_global_properties(self, backend):
        "Data globalructor should correctly set attributes."
        g = op2.Global((2, 2), [1] * 4, 'double', 'bar')
        assert g.dim == (2, 2) and g.dtype == np.float64 and g.name == 'bar' \
            and g.data.sum() == 4

    def test_global_setter(self, backend, g):
        "Setter attribute on data should correct set data value."
        g.data = 2
        assert g.data.sum() == 2

    def test_global_setter_malformed_data(self, backend, g):
        "Setter attribute should reject malformed data."
        with pytest.raises(exceptions.DataValueError):
            g.data = [1, 2]

    def test_global_eq(self, backend):
        "Globals should compare equal when having the same dim and data."
        assert op2.Global(1, [1.0]) == op2.Global(1, [1.0])
        assert not op2.Global(1, [1.0]) != op2.Global(1, [1.0])

    def test_global_ne_dim(self, backend):
        "Globals should not compare equal when having different dims."
        assert op2.Global(1) != op2.Global(2)
        assert not op2.Global(1) == op2.Global(2)

    def test_global_ne_data(self, backend):
        "Globals should not compare equal when having different data."
        assert op2.Global(1, [1.0]) != op2.Global(1, [2.0])
        assert not op2.Global(1, [1.0]) == op2.Global(1, [2.0])

    def test_global_iter(self, backend, g):
        "Global should be iterable and yield self."
        for g_ in g:
            assert g_ is g

    def test_global_len(self, backend, g):
        "Global len should be 1."
        assert len(g) == 1

    def test_global_repr(self, backend):
        "Global repr should produce a Global object when eval'd."
        from pyop2.op2 import Global  # noqa: needed by eval
        from numpy import array, dtype  # noqa: needed by eval
        g = op2.Global(1, 1, 'double')
        assert isinstance(eval(repr(g)), op2.Global)

    def test_global_str(self, backend):
        "Global should have the expected string representation."
        g = op2.Global(1, 1, 'double')
        s = "OP2 Global Argument: %s with dim %s and value %s" \
            % (g.name, g.dim, g.data)
        assert str(g) == s

    @pytest.mark.parametrize("mode", [op2.RW, op2.WRITE])
    def test_global_arg_illegal_mode(self, backend, g, mode):
        """Global __call__ should not allow illegal access modes."""
        with pytest.raises(exceptions.ModeValueError):
            g(mode)

    def test_global_arg_ignore_map(self, backend, g, m_iterset_toset):
        """Global __call__ should ignore the optional second argument."""
        assert g(op2.READ, m_iterset_toset).map is None


class TestMapAPI:

    """
    Map API unit tests
    """

    def test_map_illegal_iterset(self, backend, set):
        "Map iterset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Map('illegalset', set, 1, [])

    def test_map_illegal_toset(self, backend, set):
        "Map toset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Map(set, 'illegalset', 1, [])

    def test_map_illegal_arity(self, backend, set):
        "Map arity should be int."
        with pytest.raises(exceptions.ArityTypeError):
            op2.Map(set, set, 'illegalarity', [])

    def test_map_illegal_arity_tuple(self, backend, set):
        "Map arity should not be a tuple."
        with pytest.raises(exceptions.ArityTypeError):
            op2.Map(set, set, (2, 2), [])

    def test_map_illegal_name(self, backend, set):
        "Map name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Map(set, set, 1, [], name=2)

    def test_map_illegal_dtype(self, backend, set):
        "Illegal data type should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Map(set, set, 1, 'abcdefg')

    def test_map_illegal_length(self, backend, iterset, toset):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Map(iterset, toset, 1, [1] * (iterset.size + 1))

    def test_map_convert_float_int(self, backend, iterset, toset):
        "Float data should be implicitely converted to int."
        m = op2.Map(iterset, toset, 1, [1.5] * iterset.size)
        assert m.values.dtype == np.int32 and m.values.sum() == iterset.size

    def test_map_reshape(self, backend, iterset, toset):
        "Data should be reshaped according to arity."
        m = op2.Map(iterset, toset, 2, [1] * 2 * iterset.size)
        assert m.arity == 2 and m.values.shape == (iterset.size, 2)

    def test_map_split(self, backend, m_iterset_toset):
        "Splitting a Map should yield a tuple with self"
        for m in m_iterset_toset.split:
            m == m_iterset_toset

    def test_map_properties(self, backend, iterset, toset):
        "Data constructor should correctly set attributes."
        m = op2.Map(iterset, toset, 2, [1] * 2 * iterset.size, 'bar')
        assert (m.iterset == iterset and m.toset == toset and m.arity == 2 and
                m.arities == (2,) and m.arange == (0, 2) and
                m.values.sum() == 2 * iterset.size and m.name == 'bar')

    def test_map_indexing(self, backend, m_iterset_toset):
        "Indexing a map should create an appropriate Arg"
        assert m_iterset_toset[0].idx == 0

    def test_map_slicing(self, backend, m_iterset_toset):
        "Slicing a map is not allowed"
        with pytest.raises(NotImplementedError):
            m_iterset_toset[:]

    def test_map_eq(self, backend, m_iterset_toset):
        """Map equality is identity."""
        mcopy = op2.Map(m_iterset_toset.iterset, m_iterset_toset.toset,
                        m_iterset_toset.arity, m_iterset_toset.values)
        assert m_iterset_toset != mcopy
        assert not m_iterset_toset == mcopy
        assert mcopy == mcopy

    def test_map_ne_iterset(self, backend, m_iterset_toset):
        """Maps that have copied but not equal iteration sets are not equal."""
        mcopy = op2.Map(op2.Set(m_iterset_toset.iterset.size),
                        m_iterset_toset.toset, m_iterset_toset.arity,
                        m_iterset_toset.values)
        assert m_iterset_toset != mcopy
        assert not m_iterset_toset == mcopy

    def test_map_ne_toset(self, backend, m_iterset_toset):
        """Maps that have copied but not equal to sets are not equal."""
        mcopy = op2.Map(m_iterset_toset.iterset, op2.Set(m_iterset_toset.toset.size),
                        m_iterset_toset.arity, m_iterset_toset.values)
        assert m_iterset_toset != mcopy
        assert not m_iterset_toset == mcopy

    def test_map_ne_arity(self, backend, m_iterset_toset):
        """Maps that have different arities are not equal."""
        mcopy = op2.Map(m_iterset_toset.iterset, m_iterset_toset.toset,
                        m_iterset_toset.arity * 2, list(m_iterset_toset.values) * 2)
        assert m_iterset_toset != mcopy
        assert not m_iterset_toset == mcopy

    def test_map_ne_values(self, backend, m_iterset_toset):
        """Maps that have different values are not equal."""
        m2 = op2.Map(m_iterset_toset.iterset, m_iterset_toset.toset,
                     m_iterset_toset.arity, m_iterset_toset.values.copy())
        m2.values[0] = 2
        assert m_iterset_toset != m2
        assert not m_iterset_toset == m2

    def test_map_iter(self, backend, m_iterset_toset):
        "Map should be iterable and yield self."
        for m_ in m_iterset_toset:
            assert m_ is m_iterset_toset

    def test_map_len(self, backend, m_iterset_toset):
        "Map len should be 1."
        assert len(m_iterset_toset) == 1

    def test_map_repr(self, backend, m_iterset_toset):
        "Map should have the expected repr."
        r = "Map(%r, %r, %r, None, %r)" % (m_iterset_toset.iterset, m_iterset_toset.toset,
                                           m_iterset_toset.arity, m_iterset_toset.name)
        assert repr(m_iterset_toset) == r

    def test_map_str(self, backend, m_iterset_toset):
        "Map should have the expected string representation."
        s = "OP2 Map: %s from (%s) to (%s) with arity %s" \
            % (m_iterset_toset.name, m_iterset_toset.iterset, m_iterset_toset.toset, m_iterset_toset.arity)
        assert str(m_iterset_toset) == s


class TestMixedMapAPI:

    """
    MixedMap API unit tests
    """

    def test_mixed_map_illegal_arg(self, backend):
        "Map iterset should be Set."
        with pytest.raises(TypeError):
            op2.MixedMap('illegalarg')

    def test_mixed_map_split(self, backend, maps):
        """Constructing a MixedDat from an iterable of Maps should leave them
        unchanged."""
        mmap = op2.MixedMap(maps)
        assert mmap.split == maps
        for i, m in enumerate(maps):
            assert mmap.split[i] == m
        assert mmap.split[:-1] == tuple(mmap)[:-1]

    def test_mixed_map_nonunique_itset(self, backend, m_iterset_toset, m_set_toset):
        "Map toset should be Set."
        with pytest.raises(exceptions.MapTypeError):
            op2.MixedMap((m_iterset_toset, m_set_toset))

    def test_mixed_map_iterset(self, backend, mmap):
        "MixedMap iterset should return the common iterset of all Maps."
        for m in mmap:
            assert mmap.iterset == m.iterset

    def test_mixed_map_toset(self, backend, mmap):
        "MixedMap toset should return a MixedSet of the Map tosets."
        assert mmap.toset == op2.MixedSet(m.toset for m in mmap)

    def test_mixed_map_arity(self, backend, mmap):
        "MixedMap arity should return the sum of the Map arities."
        assert mmap.arity == sum(m.arity for m in mmap)

    def test_mixed_map_arities(self, backend, mmap):
        "MixedMap arities should return a tuple of the Map arities."
        assert mmap.arities == tuple(m.arity for m in mmap)

    def test_mixed_map_arange(self, backend, mmap):
        "MixedMap arities should return a tuple of the Map arities."
        assert mmap.arange == (0,) + tuple(np.cumsum(mmap.arities))

    def test_mixed_map_values(self, backend, mmap):
        "MixedMap values should return a tuple of the Map values."
        assert all((v == m.values).all() for v, m in zip(mmap.values, mmap))

    def test_mixed_map_values_with_halo(self, backend, mmap):
        "MixedMap values_with_halo should return a tuple of the Map values."
        assert all((v == m.values_with_halo).all() for v, m in zip(mmap.values_with_halo, mmap))

    def test_mixed_map_name(self, backend, mmap):
        "MixedMap name should return a tuple of the Map names."
        assert mmap.name == tuple(m.name for m in mmap)

    def test_mixed_map_offset(self, backend, mmap):
        "MixedMap offset should return a tuple of the Map offsets."
        assert mmap.offset == tuple(m.offset for m in mmap)

    def test_mixed_map_iter(self, backend, maps):
        "MixedMap should be iterable and yield the Maps."
        assert tuple(m for m in op2.MixedMap(maps)) == maps

    def test_mixed_map_len(self, backend, maps):
        """MixedMap should have length equal to the number of contained Maps."""
        assert len(op2.MixedMap(maps)) == len(maps)

    def test_mixed_map_eq(self, backend, maps):
        "MixedMaps created from the same Maps should compare equal."
        assert op2.MixedMap(maps) == op2.MixedMap(maps)
        assert not op2.MixedMap(maps) != op2.MixedMap(maps)

    def test_mixed_map_ne(self, backend, maps):
        "MixedMaps created from different Maps should not compare equal."
        mm1 = op2.MixedMap((maps[0], maps[1]))
        mm2 = op2.MixedMap((maps[1], maps[0]))
        assert mm1 != mm2
        assert not mm1 == mm2

    def test_mixed_map_ne_map(self, backend, maps):
        "A MixedMap should not compare equal to a Map."
        assert op2.MixedMap(maps) != maps[0]
        assert not op2.MixedMap(maps) == maps[0]

    def test_mixed_map_repr(self, backend, mmap):
        "MixedMap should have the expected repr."
        # Note: We can't actually reproduce a MixedMap from its repr because
        # the iteration sets will not be identical, which is checked in the
        # constructor
        assert repr(mmap) == "MixedMap(%r)" % (mmap.split,)

    def test_mixed_map_str(self, backend, mmap):
        "MixedMap should have the expected string representation."
        assert str(mmap) == "OP2 MixedMap composed of Maps: %s" % (mmap.split,)


class TestIterationSpaceAPI:

    """
    IterationSpace API unit tests
    """

    def test_iteration_space_illegal_iterset(self, backend, set):
        "IterationSpace iterset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            base.IterationSpace('illegalset', 1)

    def test_iteration_space_illegal_block_shape(self, backend, set):
        "IterationSpace extents should be int or int tuple."
        with pytest.raises(TypeError):
            base.IterationSpace(set, 'illegalextents')

    def test_iteration_space_illegal_extents_tuple(self, backend, set):
        "IterationSpace extents should be int or int tuple."
        with pytest.raises(TypeError):
            base.IterationSpace(set, (1, 'illegalextents'))

    def test_iteration_space_iter(self, backend, set):
        "Iterating an empty IterationSpace should yield an empty shape."
        for i, j, shape, offset in base.IterationSpace(set):
            assert i == 0 and j == 0 and shape == () and offset == (0, 0)

    def test_iteration_space_eq(self, backend, set):
        """IterationSpaces should compare equal if defined on the same Set."""
        assert base.IterationSpace(set) == base.IterationSpace(set)
        assert not base.IterationSpace(set) != base.IterationSpace(set)

    def test_iteration_space_ne_set(self, backend):
        """IterationSpaces should not compare equal if defined on different
        Sets."""
        assert base.IterationSpace(op2.Set(3)) != base.IterationSpace(op2.Set(3))
        assert not base.IterationSpace(op2.Set(3)) == base.IterationSpace(op2.Set(3))

    def test_iteration_space_ne_block_shape(self, backend, set):
        """IterationSpaces should not compare equal if defined with different
        block shapes."""
        assert base.IterationSpace(set, (((3,),),)) != base.IterationSpace(set, (((2,),),))
        assert not base.IterationSpace(set, (((3,),),)) == base.IterationSpace(set, (((2,),),))

    def test_iteration_space_repr(self, backend, set):
        """IterationSpace repr should produce a IterationSpace object when
        eval'd."""
        from pyop2.op2 import Set  # noqa: needed by eval
        from pyop2.base import IterationSpace  # noqa: needed by eval
        m = IterationSpace(set)
        assert isinstance(eval(repr(m)), IterationSpace)

    def test_iteration_space_str(self, backend, set):
        "IterationSpace should have the expected string representation."
        m = base.IterationSpace(set)
        s = "OP2 Iteration Space: %s with extents %s" % (m.iterset, m.extents)
        assert str(m) == s


class TestKernelAPI:

    """
    Kernel API unit tests
    """

    def test_kernel_illegal_name(self, backend):
        "Kernel name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Kernel("", name=2)

    def test_kernel_properties(self, backend):
        "Kernel constructor should correctly set attributes."
        k = op2.Kernel("", 'foo')
        assert k.name == 'foo'

    def test_kernel_repr(self, backend, set):
        "Kernel should have the expected repr."
        k = op2.Kernel("int foo() { return 0; }", 'foo')
        assert repr(k) == 'Kernel("""%s""", %r)' % (k.code(), k.name)

    def test_kernel_str(self, backend, set):
        "Kernel should have the expected string representation."
        k = op2.Kernel("int foo() { return 0; }", 'foo')
        assert str(k) == "OP2 Kernel: %s" % k.name


class TestParLoopAPI:

    """
    ParLoop API unit tests
    """

    def test_illegal_kernel(self, backend, set, dat, m_iterset_toset):
        """The first ParLoop argument has to be of type op2.Kernel."""
        with pytest.raises(exceptions.KernelTypeError):
            op2.par_loop('illegal_kernel', set, dat(op2.READ, m_iterset_toset))

    def test_illegal_iterset(self, backend, dat, m_iterset_toset):
        """The first ParLoop argument has to be of type op2.Kernel."""
        with pytest.raises(exceptions.SetTypeError):
            op2.par_loop(op2.Kernel("", "k"), 'illegal_set',
                         dat(op2.READ, m_iterset_toset))

    def test_illegal_dat_iterset(self, backend):
        """ParLoop should reject a Dat argument using a different iteration
        set from the par_loop's."""
        set1 = op2.Set(2)
        set2 = op2.Set(3)
        dset1 = op2.DataSet(set1, 1)
        dat = op2.Dat(dset1)
        map = op2.Map(set2, set1, 1, [0, 0, 0])
        kernel = op2.Kernel("void k() { }", "k")
        with pytest.raises(exceptions.MapValueError):
            base.ParLoop(kernel, set1, dat(op2.READ, map))

    def test_illegal_mat_iterset(self, backend, skip_opencl, sparsity):
        """ParLoop should reject a Mat argument using a different iteration
        set from the par_loop's."""
        set1 = op2.Set(2)
        m = op2.Mat(sparsity)
        rmap, cmap = sparsity.maps[0]
        kernel = op2.Kernel("void k() { }", "k")
        with pytest.raises(exceptions.MapValueError):
            op2.par_loop(kernel, set1,
                         m(op2.INC, (rmap[op2.i[0]], cmap[op2.i[1]])))

    def test_empty_map_and_iterset(self, backend):
        """If the iterset of the ParLoop is zero-sized, it should not matter if
        a map defined on it has no values."""
        s1 = op2.Set(0)
        s2 = op2.Set(10)
        m = op2.Map(s1, s2, 3)
        d = op2.Dat(s2 ** 1, [0] * 10, dtype=int)
        k = op2.Kernel("void k(int *x) {}", "k")
        op2.par_loop(k, s1, d(op2.READ, m[0]))


class TestSolverAPI:

    """
    Test the Solver API.
    """

    def test_solver_defaults(self, backend):
        s = op2.Solver()
        assert s.parameters == base.DEFAULT_SOLVER_PARAMETERS

    def test_set_options_with_params(self, backend):
        params = {'ksp_type': 'gmres',
                  'ksp_max_it': 25}
        s = op2.Solver(params)
        assert s.parameters['ksp_type'] == 'gmres' \
            and s.parameters['ksp_max_it'] == 25

    def test_set_options_with_kwargs(self, backend):
        s = op2.Solver(ksp_type='gmres', ksp_max_it=25)
        assert s.parameters['ksp_type'] == 'gmres' \
            and s.parameters['ksp_max_it'] == 25

    def test_update_parameters(self, backend):
        s = op2.Solver()
        params = {'ksp_type': 'gmres',
                  'ksp_max_it': 25}
        s.update_parameters(params)
        assert s.parameters['ksp_type'] == 'gmres' \
            and s.parameters['ksp_max_it'] == 25

    def test_set_params_and_kwargs_illegal(self, backend):
        params = {'ksp_type': 'gmres',
                  'ksp_max_it': 25}
        with pytest.raises(RuntimeError):
            op2.Solver(params, ksp_type='cgs')

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
