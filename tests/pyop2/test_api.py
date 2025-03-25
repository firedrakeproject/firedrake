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

from pyop2 import exceptions, op2
from pyop2.mpi import COMM_WORLD


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
def mds(dtoset, set):
    return op2.MixedDataSet((dtoset, set))


# pytest doesn't currently support using fixtures are paramters to tests
# or other fixtures. We have to work around that by requesting fixtures
# by name
@pytest.fixture(params=[('mds', 'mds', 'mmap', 'mmap'),
                        ('mds', 'dtoset', 'mmap', 'm_iterset_toset'),
                        ('dtoset', 'mds', 'm_iterset_toset', 'mmap')])
def ms(request):
    rds, cds, rmm, cmm = [request.getfixturevalue(p) for p in request.param]
    return op2.Sparsity((rds, cds), {(i, j): [(rm, cm, None)] for i, rm in enumerate(rmm) for j, cm in enumerate(cmm)})


@pytest.fixture
def sparsity(m_iterset_toset, dtoset):
    return op2.Sparsity((dtoset, dtoset), [(m_iterset_toset, m_iterset_toset, None)])


@pytest.fixture
def mat(sparsity):
    return op2.Mat(sparsity)


@pytest.fixture
def diag_mat(toset):
    _d = toset ** 1
    _m = op2.Map(toset, toset, 1, np.arange(toset.size))
    return op2.Mat(op2.Sparsity((_d, _d), [(_m, _m, None)]))


@pytest.fixture
def mmat(ms):
    return op2.Mat(ms)


@pytest.fixture
def g():
    return op2.Global(1, 1, comm=COMM_WORLD)


class TestClassAPI:

    """Do PyOP2 classes behave like normal classes?"""

    def test_isinstance(self, set, dat):
        "isinstance should behave as expected."
        assert isinstance(set, op2.Set)
        assert isinstance(dat, op2.Dat)
        assert not isinstance(set, op2.Dat)
        assert not isinstance(dat, op2.Set)

    def test_issubclass(self, set, dat):
        "issubclass should behave as expected"
        assert issubclass(type(set), op2.Set)
        assert issubclass(type(dat), op2.Dat)
        assert not issubclass(type(set), op2.Dat)
        assert not issubclass(type(dat), op2.Set)


class TestSetAPI:

    """
    Set API unit tests
    """

    def test_set_illegal_size(self):
        "Set size should be int."
        with pytest.raises(exceptions.SizeTypeError):
            op2.Set('illegalsize')

    def test_set_illegal_name(self):
        "Set name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Set(1, 2)

    def test_set_iter(self, set):
        "Set should be iterable and yield self."
        for s in set:
            assert s is set

    def test_set_len(self, set):
        "Set len should be 1."
        assert len(set) == 1

    def test_set_repr(self, set):
        "Set repr should produce a Set object when eval'd."
        from pyop2.op2 import Set  # noqa: needed by eval
        assert isinstance(eval(repr(set)), op2.Set)

    def test_set_str(self, set):
        "Set should have the expected string representation."
        assert str(set) == "OP2 Set: %s with size %s" % (set.name, set.size)

    def test_set_eq(self, set):
        "The equality test for sets is identity, not attribute equality"
        assert set == set
        assert not set != set

    def test_dset_in_set(self, set, dset):
        "The in operator should indicate compatibility of DataSet and Set"
        assert dset in set

    def test_dset_not_in_set(self, dset):
        "The in operator should indicate incompatibility of DataSet and Set"
        assert dset not in op2.Set(5, 'bar')

    def test_set_exponentiation_builds_dset(self, set):
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
    def test_illegal_layers_arg(self, set):
        """Must pass at least 2 as a layers argument"""
        with pytest.raises(exceptions.SizeTypeError):
            op2.ExtrudedSet(set, 1)

    def test_illegal_set_arg(self):
        """Extuded Set should be build on a Set"""
        with pytest.raises(TypeError):
            op2.ExtrudedSet(1, 3)

    def test_set_compatiblity(self, set, iterset):
        """The set an extruded set was built on should be contained in it"""
        e = op2.ExtrudedSet(set, 5)
        assert set in e
        assert iterset not in e

    def test_iteration_compatibility(self, iterset, m_iterset_toset, m_iterset_set, dats):
        """It should be possible to iterate over an extruded set reading dats
           defined on the base set (indirectly)."""
        e = op2.ExtrudedSet(iterset, 5)
        k = op2.Kernel('static void k() { }', 'k')
        dat1, dat2 = dats
        op2.par_loop(k, e, dat1(op2.READ, m_iterset_toset))
        op2.par_loop(k, e, dat2(op2.READ, m_iterset_set))

    def test_iteration_incompatibility(self, set, m_iterset_toset, dat):
        """It should not be possible to iteratve over an extruded set reading
           dats not defined on the base set (indirectly)."""
        e = op2.ExtrudedSet(set, 5)
        k = op2.Kernel('static void k() { }', 'k')
        with pytest.raises(exceptions.MapValueError):
            op2.ParLoop(k, e, dat(op2.READ, m_iterset_toset))


class TestSubsetAPI:
    """
    Subset API unit tests
    """

    def test_illegal_set_arg(self):
        "The subset constructor checks arguments."
        with pytest.raises(TypeError):
            op2.Subset("fail", [0, 1])

    def test_out_of_bounds_index(self, set):
        "The subset constructor checks indices are correct."
        with pytest.raises(exceptions.SubsetIndexOutOfBounds):
            op2.Subset(set, list(range(set.total_size + 1)))

    def test_invalid_index(self, set):
        "The subset constructor checks indices are correct."
        with pytest.raises(exceptions.SubsetIndexOutOfBounds):
            op2.Subset(set, [-1])

    def test_empty_subset(self, set):
        "Subsets can be empty."
        ss = op2.Subset(set, [])
        assert len(ss.indices) == 0

    def test_index_construction(self, set):
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

    def test_indices_duplicate_removed(self, set):
        "The subset constructor voids duplicate indices)"
        ss = op2.Subset(set, [0, 0, 1, 1])
        assert np.sum(ss.indices == 0) == 1
        assert np.sum(ss.indices == 1) == 1

    def test_indices_sorted(self, set):
        "The subset constructor sorts indices)"
        ss = op2.Subset(set, [0, 4, 1, 2, 3])
        assert_equal(ss.indices, list(range(5)))

        ss2 = op2.Subset(set, list(range(5)))
        assert_equal(ss.indices, ss2.indices)


class TestMixedSetAPI:

    """
    MixedSet API unit tests
    """

    def test_mixed_set_illegal_set(self):
        "MixedSet sets should be of type Set."
        with pytest.raises(TypeError):
            op2.MixedSet(('foo', 'bar'))

    def test_mixed_set_getitem(self, sets):
        "MixedSet should return the corresponding Set when indexed."
        mset = op2.MixedSet(sets)
        for i, s in enumerate(sets):
            assert mset[i] == s

    def test_mixed_set_split(self, sets):
        "MixedSet split should return a tuple of the Sets."
        assert op2.MixedSet(sets).split == sets

    def test_mixed_set_core_size(self, mset):
        "MixedSet core_size should return the sum of the Set core_sizes."
        assert mset.core_size == sum(s.core_size for s in mset)

    def test_mixed_set_size(self, mset):
        "MixedSet size should return the sum of the Set sizes."
        assert mset.size == sum(s.size for s in mset)

    def test_mixed_set_total_size(self, mset):
        "MixedSet total_size should return the sum of the Set total_sizes."
        assert mset.total_size == sum(s.total_size for s in mset)

    def test_mixed_set_sizes(self, mset):
        "MixedSet sizes should return a tuple of the Set sizes."
        assert mset.sizes == (mset.core_size, mset.size, mset.total_size)

    def test_mixed_set_name(self, mset):
        "MixedSet name should return a tuple of the Set names."
        assert mset.name == tuple(s.name for s in mset)

    def test_mixed_set_halo(self, mset):
        "MixedSet halo should be None when running sequentially."
        assert mset.halo is None

    def test_mixed_set_layers(self, mset):
        "MixedSet layers should return the layers of the first Set."
        assert mset.layers == mset[0].layers

    def test_mixed_set_layers_must_match(self, sets):
        "All components of a MixedSet must have the same number of layers."
        sets = [op2.ExtrudedSet(s, layers=i+4) for i, s in enumerate(sets)]
        with pytest.raises(AssertionError):
            op2.MixedSet(sets)

    def test_mixed_set_iter(self, mset, sets):
        "MixedSet should be iterable and yield the Sets."
        assert tuple(s for s in mset) == sets

    def test_mixed_set_len(self, sets):
        "MixedSet should have length equal to the number of contained Sets."
        assert len(op2.MixedSet(sets)) == len(sets)

    def test_mixed_set_pow_int(self, mset):
        "MixedSet should implement ** operator returning a MixedDataSet."
        assert mset ** 1 == op2.MixedDataSet([s ** 1 for s in mset])

    def test_mixed_set_pow_seq(self, mset):
        "MixedSet should implement ** operator returning a MixedDataSet."
        assert mset ** ((1,) * len(mset)) == op2.MixedDataSet([s ** 1 for s in mset])

    def test_mixed_set_pow_gen(self, mset):
        "MixedSet should implement ** operator returning a MixedDataSet."
        assert mset ** (1 for _ in mset) == op2.MixedDataSet([s ** 1 for s in mset])

    def test_mixed_set_eq(self, sets):
        "MixedSets created from the same Sets should compare equal."
        assert op2.MixedSet(sets) == op2.MixedSet(sets)
        assert not op2.MixedSet(sets) != op2.MixedSet(sets)

    def test_mixed_set_ne(self, set, iterset, toset):
        "MixedSets created from different Sets should not compare equal."
        assert op2.MixedSet((set, iterset, toset)) != op2.MixedSet((set, toset, iterset))
        assert not op2.MixedSet((set, iterset, toset)) == op2.MixedSet((set, toset, iterset))

    def test_mixed_set_ne_set(self, sets):
        "A MixedSet should not compare equal to a Set."
        assert op2.MixedSet(sets) != sets[0]
        assert not op2.MixedSet(sets) == sets[0]

    def test_mixed_set_repr(self, mset):
        "MixedSet repr should produce a MixedSet object when eval'd."
        from pyop2.op2 import Set, MixedSet  # noqa: needed by eval
        assert isinstance(eval(repr(mset)), op2.MixedSet)

    def test_mixed_set_str(self, mset):
        "MixedSet should have the expected string representation."
        assert str(mset) == "OP2 MixedSet composed of Sets: %s" % (mset._sets,)


class TestDataSetAPI:
    """
    DataSet API unit tests
    """

    def test_dset_illegal_dim(self, iterset):
        "DataSet dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.DataSet(iterset, 'illegaldim')

    def test_dset_illegal_dim_tuple(self, iterset):
        "DataSet dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.DataSet(iterset, (1, 'illegaldim'))

    def test_dset_illegal_name(self, iterset):
        "DataSet name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.DataSet(iterset, 1, 2)

    def test_dset_default_dim(self, iterset):
        "DataSet constructor should default dim to (1,)."
        assert op2.DataSet(iterset).dim == (1,)

    def test_dset_dim(self, iterset):
        "DataSet constructor should create a dim tuple."
        s = op2.DataSet(iterset, 1)
        assert s.dim == (1,)

    def test_dset_dim_list(self, iterset):
        "DataSet constructor should create a dim tuple from a list."
        s = op2.DataSet(iterset, [2, 3])
        assert s.dim == (2, 3)

    def test_dset_iter(self, dset):
        "DataSet should be iterable and yield self."
        for s in dset:
            assert s is dset

    def test_dset_len(self, dset):
        "DataSet len should be 1."
        assert len(dset) == 1

    def test_dset_repr(self, dset):
        "DataSet repr should produce a Set object when eval'd."
        from pyop2.op2 import Set, DataSet  # noqa: needed by eval
        assert isinstance(eval(repr(dset)), op2.DataSet)

    def test_dset_str(self, dset):
        "DataSet should have the expected string representation."
        assert str(dset) == "OP2 DataSet: %s on set %s, with dim %s, %s" \
            % (dset.name, dset.set, dset.dim, dset._apply_local_global_filter)

    def test_dset_eq(self, dset):
        "The equality test for DataSets is same dim and same set"
        dsetcopy = op2.DataSet(dset.set, dset.dim)
        assert dsetcopy == dset
        assert not dsetcopy != dset

    def test_dset_ne_set(self, dset):
        "DataSets with the same dim but different Sets are not equal."
        dsetcopy = op2.DataSet(op2.Set(dset.set.size), dset.dim)
        assert dsetcopy != dset
        assert not dsetcopy == dset

    def test_dset_ne_dim(self, dset):
        "DataSets with the same Set but different dims are not equal."
        dsetcopy = op2.DataSet(dset.set, tuple(d + 1 for d in dset.dim))
        assert dsetcopy != dset
        assert not dsetcopy == dset

    def test_dat_in_dset(self, dset):
        "The in operator should indicate compatibility of DataSet and Set"
        assert op2.Dat(dset) in dset

    def test_dat_not_in_dset(self, dset):
        "The in operator should indicate incompatibility of DataSet and Set"
        assert op2.Dat(dset) not in op2.DataSet(op2.Set(5, 'bar'))


class TestMixedDataSetAPI:
    """
    MixedDataSet API unit tests
    """

    @pytest.mark.parametrize('arg', ['illegalarg', (set, 'illegalarg'),
                                     iter((set, 'illegalarg'))])
    def test_mixed_dset_illegal_arg(self, arg):
        """Constructing a MixedDataSet from anything other than a MixedSet or
        an iterable of Sets and/or DataSets should fail."""
        with pytest.raises(TypeError):
            op2.MixedDataSet(arg)

    @pytest.mark.parametrize('dims', ['illegaldim', (1, 2, 'illegaldim')])
    def test_mixed_dset_dsets_illegal_dims(self, dsets, dims):
        """When constructing a MixedDataSet from an iterable of DataSets it is
        an error to specify dims."""
        with pytest.raises((TypeError, ValueError)):
            op2.MixedDataSet(dsets, dims)

    def test_mixed_dset_dsets_dims(self, dsets):
        """When constructing a MixedDataSet from an iterable of DataSets it is
        an error to specify dims."""
        with pytest.raises(TypeError):
            op2.MixedDataSet(dsets, 1)

    def test_mixed_dset_upcast_sets(self, msets, mset):
        """Constructing a MixedDataSet from an iterable/iterator of Sets or
        MixedSet should upcast."""
        assert op2.MixedDataSet(msets) == mset ** 1

    def test_mixed_dset_sets_and_dsets(self, set, dset):
        """Constructing a MixedDataSet from an iterable with a mixture of
        Sets and DataSets should upcast the Sets."""
        assert op2.MixedDataSet((set, dset)).split == (set ** 1, dset)

    def test_mixed_dset_sets_and_dsets_gen(self, set, dset):
        """Constructing a MixedDataSet from an iterable with a mixture of
        Sets and DataSets should upcast the Sets."""
        assert op2.MixedDataSet(iter((set, dset))).split == (set ** 1, dset)

    def test_mixed_dset_dims_default_to_one(self, msets, mset):
        """Constructing a MixedDataSet from an interable/iterator of Sets or
        MixedSet without dims should default them to 1."""
        assert op2.MixedDataSet(msets).dim == ((1,),) * len(mset)

    def test_mixed_dset_dims_int(self, msets, mset):
        """Construct a MixedDataSet from an iterator/iterable of Sets and a
        MixedSet with dims as an int."""
        assert op2.MixedDataSet(msets, 2).dim == ((2,),) * len(mset)

    def test_mixed_dset_dims_gen(self, msets, mset):
        """Construct a MixedDataSet from an iterator/iterable of Sets and a
        MixedSet with dims as a generator."""
        dims = (2 for _ in mset)
        assert op2.MixedDataSet(msets, dims).dim == ((2,),) * len(mset)

    def test_mixed_dset_dims_iterable(self, msets):
        """Construct a MixedDataSet from an iterator/iterable of Sets and a
        MixedSet with dims as an iterable."""
        dims = ((2,), (2, 2), (1,))
        assert op2.MixedDataSet(msets, dims).dim == dims

    def test_mixed_dset_dims_mismatch(self, msets, sets):
        """Constructing a MixedDataSet from an iterable/iterator of Sets and a
        MixedSet with mismatching number of dims should raise ValueError."""
        with pytest.raises(ValueError):
            op2.MixedDataSet(msets, list(range(1, len(sets))))

    def test_mixed_dset_getitem(self, mdset):
        "MixedDataSet should return the corresponding DataSet when indexed."
        for i, ds in enumerate(mdset):
            assert mdset[i] == ds

    def test_mixed_dset_split(self, dsets):
        "MixedDataSet split should return a tuple of the DataSets."
        assert op2.MixedDataSet(dsets).split == dsets

    def test_mixed_dset_dim(self, mdset):
        "MixedDataSet dim should return a tuple of the DataSet dims."
        assert mdset.dim == tuple(s.dim for s in mdset)

    def test_mixed_dset_cdim(self, mdset):
        "MixedDataSet cdim should return the sum of the DataSet cdims."
        assert mdset.cdim == sum(s.cdim for s in mdset)

    def test_mixed_dset_name(self, mdset):
        "MixedDataSet name should return a tuple of the DataSet names."
        assert mdset.name == tuple(s.name for s in mdset)

    def test_mixed_dset_set(self, mset):
        "MixedDataSet set should return a MixedSet."
        assert op2.MixedDataSet(mset).set == mset

    def test_mixed_dset_iter(self, mdset, dsets):
        "MixedDataSet should be iterable and yield the DataSets."
        assert tuple(s for s in mdset) == dsets

    def test_mixed_dset_len(self, dsets):
        """MixedDataSet should have length equal to the number of contained
        DataSets."""
        assert len(op2.MixedDataSet(dsets)) == len(dsets)

    def test_mixed_dset_eq(self, dsets):
        "MixedDataSets created from the same DataSets should compare equal."
        assert op2.MixedDataSet(dsets) == op2.MixedDataSet(dsets)
        assert not op2.MixedDataSet(dsets) != op2.MixedDataSet(dsets)

    def test_mixed_dset_ne(self, dset, diterset, dtoset):
        "MixedDataSets created from different DataSets should not compare equal."
        mds1 = op2.MixedDataSet((dset, diterset, dtoset))
        mds2 = op2.MixedDataSet((dset, dtoset, diterset))
        assert mds1 != mds2
        assert not mds1 == mds2

    def test_mixed_dset_ne_dset(self, diterset, dtoset):
        "MixedDataSets should not compare equal to a scalar DataSet."
        assert op2.MixedDataSet((diterset, dtoset)) != diterset
        assert not op2.MixedDataSet((diterset, dtoset)) == diterset

    def test_mixed_dset_repr(self, mdset):
        "MixedDataSet repr should produce a MixedDataSet object when eval'd."
        from pyop2.op2 import Set, DataSet, MixedDataSet  # noqa: needed by eval
        assert isinstance(eval(repr(mdset)), op2.MixedDataSet)

    def test_mixed_dset_str(self, mdset):
        "MixedDataSet should have the expected string representation."
        assert str(mdset) == "OP2 MixedDataSet composed of DataSets: %s" % (mdset._dsets,)


class TestDatAPI:

    """
    Dat API unit tests
    """

    def test_dat_illegal_set(self):
        "Dat set should be DataSet."
        with pytest.raises(exceptions.DataSetTypeError):
            op2.Dat('illegalset', 1)

    def test_dat_illegal_name(self, dset):
        "Dat name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Dat(dset, name=2)

    def test_dat_initialise_data(self, dset):
        """Dat initilialised without the data should initialise data with the
        correct size and type."""
        d = op2.Dat(dset)
        assert d.data.size == dset.size * dset.cdim and d.data.dtype == np.float64

    def test_dat_initialise_data_type(self, dset):
        """Dat intiialised without the data but with specified type should
        initialise its data with the correct type."""
        d = op2.Dat(dset, dtype=np.int32)
        assert d.data.dtype == np.int32

    def test_dat_subscript(self, dat):
        """Extracting component 0 of a Dat should yield self."""
        assert dat[0] is dat

    def test_dat_illegal_subscript(self, dat):
        """Extracting component 0 of a Dat should yield self."""
        with pytest.raises(exceptions.IndexValueError):
            dat[1]

    def test_dat_arg_default_map(self, dat):
        """Dat __call__ should default the Arg map to None if not given."""
        assert dat(op2.READ).map_ is None

    def test_dat_arg_illegal_map(self, dset):
        """Dat __call__ should not allow a map with a toset other than this
        Dat's set."""
        d = op2.Dat(dset)
        set1 = op2.Set(3)
        set2 = op2.Set(2)
        to_set2 = op2.Map(set1, set2, 1, [0, 0, 0])
        with pytest.raises(exceptions.MapValueError):
            d(op2.READ, to_set2)

    def test_dat_on_set_builds_dim_one_dataset(self, set):
        """If a Set is passed as the dataset argument, it should be
        converted into a Dataset with dim=1"""
        d = op2.Dat(set)
        assert d.cdim == 1
        assert isinstance(d.dataset, op2.DataSet)
        assert d.dataset.cdim == 1

    def test_dat_dtype_type(self, dset):
        "The type of a Dat's dtype property should be a numpy.dtype."
        d = op2.Dat(dset)
        assert isinstance(d.dtype, np.dtype)
        d = op2.Dat(dset, [1.0] * dset.size * dset.cdim)
        assert isinstance(d.dtype, np.dtype)

    def test_dat_split(self, dat):
        "Splitting a Dat should yield a tuple with self"
        for d in dat.split:
            d == dat

    def test_dat_dtype(self, dset):
        "Default data type should be numpy.float64."
        d = op2.Dat(dset)
        assert d.dtype == np.double

    def test_dat_float(self, dset):
        "Data type for float data should be numpy.float64."
        d = op2.Dat(dset, [1.0] * dset.size * dset.cdim)
        assert d.dtype == np.double

    def test_dat_int(self, dset):
        "Data type for int data should be numpy.int."
        d = op2.Dat(dset, [1] * dset.size * dset.cdim)
        assert d.dtype == np.asarray(1).dtype

    def test_dat_convert_int_float(self, dset):
        "Explicit float type should override NumPy's default choice of int."
        d = op2.Dat(dset, [1] * dset.size * dset.cdim, np.double)
        assert d.dtype == np.float64

    def test_dat_convert_float_int(self, dset):
        "Explicit int type should override NumPy's default choice of float."
        d = op2.Dat(dset, [1.5] * dset.size * dset.cdim, np.int32)
        assert d.dtype == np.int32

    def test_dat_illegal_dtype(self, dset):
        "Illegal data type should raise DataTypeError."
        with pytest.raises(exceptions.DataTypeError):
            op2.Dat(dset, dtype='illegal_type')

    def test_dat_illegal_length(self, dset):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Dat(dset, [1] * (dset.size * dset.cdim + 1))

    def test_dat_reshape(self, dset):
        "Data should be reshaped according to the set's dim."
        d = op2.Dat(dset, [1.0] * dset.size * dset.cdim)
        shape = (dset.size,) + (() if dset.cdim == 1 else dset.dim)
        assert d.data.shape == shape

    def test_dat_properties(self, dset):
        "Dat constructor should correctly set attributes."
        d = op2.Dat(dset, [1] * dset.size * dset.cdim, 'double', 'bar')
        assert d.dataset.set == dset.set and d.dtype == np.float64 and \
            d.name == 'bar' and d.data.sum() == dset.size * dset.cdim

    def test_dat_iter(self, dat):
        "Dat should be iterable and yield self."
        for d in dat:
            assert d is dat

    def test_dat_len(self, dat):
        "Dat len should be 1."
        assert len(dat) == 1

    def test_dat_repr(self, dat):
        "Dat repr should produce a Dat object when eval'd."
        from pyop2.op2 import Dat, DataSet, Set  # noqa: needed by eval
        from numpy import dtype  # noqa: needed by eval
        assert isinstance(eval(repr(dat)), op2.Dat)

    def test_dat_str(self, dset):
        "Dat should have the expected string representation."
        d = op2.Dat(dset, dtype='double', name='bar')
        s = "OP2 Dat: %s on (%s) with datatype %s" \
            % (d.name, d.dataset, d.data.dtype.name)
        assert str(d) == s

    def test_dat_ro_accessor(self, dat):
        "Attempting to set values through the RO accessor should raise an error."
        x = dat.data_ro
        with pytest.raises((RuntimeError, ValueError)):
            x[0] = 1

    def test_dat_ro_write_accessor(self, dat):
        "Re-accessing the data in writeable form should be allowed."
        x = dat.data_ro
        with pytest.raises((RuntimeError, ValueError)):
            x[0] = 1
        x = dat.data
        x[0] = -100
        assert (dat.data_ro[0] == -100).all()

    def test_dat_lazy_allocation(self, dset):
        "Temporary Dats should not allocate storage until accessed."
        d = op2.Dat(dset)
        assert not d._is_allocated

    def test_dat_zero_cdim(self, set):
        "A Dat built on a DataSet with zero dim should be allowed."
        dset = set**0
        d = op2.Dat(dset)
        assert d.shape == (set.total_size, 0)
        assert d._data.size == 0
        assert d._data.shape == (set.total_size, 0)


class TestMixedDatAPI:

    """
    MixedDat API unit tests
    """

    def test_mixed_dat_illegal_arg(self):
        """Constructing a MixedDat from anything other than a MixedSet, a
        MixedDataSet or an iterable of Dats should fail."""
        with pytest.raises(exceptions.DataSetTypeError):
            op2.MixedDat('illegalarg')

    def test_mixed_dat_illegal_dtype(self, set):
        """Constructing a MixedDat from Dats of different dtype should fail."""
        with pytest.raises(exceptions.DataValueError):
            op2.MixedDat((op2.Dat(set, dtype=np.int32), op2.Dat(set)))

    def test_mixed_dat_dats(self, dats):
        """Constructing a MixedDat from an iterable of Dats should leave them
        unchanged."""
        assert op2.MixedDat(dats).split == dats

    def test_mixed_dat_dsets(self, mdset):
        """Constructing a MixedDat from an iterable of DataSets should leave
        them unchanged."""
        assert op2.MixedDat(mdset).dataset == mdset

    def test_mixed_dat_upcast_sets(self, mset):
        "Constructing a MixedDat from an iterable of Sets should upcast."
        assert op2.MixedDat(mset).dataset == op2.MixedDataSet(mset)

    def test_mixed_dat_getitem(self, mdat):
        "MixedDat should return the corresponding Dat when indexed."
        for i, d in enumerate(mdat):
            assert mdat[i] == d
        assert mdat[:-1] == tuple(mdat)[:-1]

    def test_mixed_dat_dim(self, mdset):
        "MixedDat dim should return a tuple of the DataSet dims."
        assert op2.MixedDat(mdset).dim == mdset.dim

    def test_mixed_dat_cdim(self, mdset):
        "MixedDat cdim should return a tuple of the DataSet cdims."
        assert op2.MixedDat(mdset).cdim == mdset.cdim

    def test_mixed_dat_data(self, mdat):
        "MixedDat data should return a tuple of the Dat data arrays."
        assert all((d1 == d2.data).all() for d1, d2 in zip(mdat.data, mdat))

    def test_mixed_dat_data_ro(self, mdat):
        "MixedDat data_ro should return a tuple of the Dat data_ro arrays."
        assert all((d1 == d2.data_ro).all() for d1, d2 in zip(mdat.data_ro, mdat))

    def test_mixed_dat_data_with_halos(self, mdat):
        """MixedDat data_with_halos should return a tuple of the Dat
        data_with_halos arrays."""
        assert all((d1 == d2.data_with_halos).all() for d1, d2 in zip(mdat.data_with_halos, mdat))

    def test_mixed_dat_data_ro_with_halos(self, mdat):
        """MixedDat data_ro_with_halos should return a tuple of the Dat
        data_ro_with_halos arrays."""
        assert all((d1 == d2.data_ro_with_halos).all() for d1, d2 in zip(mdat.data_ro_with_halos, mdat))

    def test_mixed_dat_needs_halo_update(self, mdat):
        """MixedDat needs_halo_update should indicate if at least one contained
        Dat needs a halo update."""
        assert mdat.halo_valid
        mdat[0].halo_valid = False
        assert not mdat.halo_valid

    def test_mixed_dat_needs_halo_update_setter(self, mdat):
        """Setting MixedDat needs_halo_update should set the property for all
        contained Dats."""
        assert mdat.halo_valid
        mdat.halo_valid = False
        assert not any(d.halo_valid for d in mdat)

    def test_mixed_dat_iter(self, mdat, dats):
        "MixedDat should be iterable and yield the Dats."
        assert tuple(s for s in mdat) == dats

    def test_mixed_dat_len(self, dats):
        """MixedDat should have length equal to the number of contained Dats."""
        assert len(op2.MixedDat(dats)) == len(dats)

    def test_mixed_dat_eq(self, dats):
        "MixedDats created from the same Dats should compare equal."
        assert op2.MixedDat(dats) == op2.MixedDat(dats)
        assert not op2.MixedDat(dats) != op2.MixedDat(dats)

    def test_mixed_dat_ne(self, dats):
        "MixedDats created from different Dats should not compare equal."
        mdat1 = op2.MixedDat(dats)
        mdat2 = op2.MixedDat(reversed(dats))
        assert mdat1 != mdat2
        assert not mdat1 == mdat2

    def test_mixed_dat_ne_dat(self, dats):
        "A MixedDat should not compare equal to a Dat."
        assert op2.MixedDat(dats) != dats[0]
        assert not op2.MixedDat(dats) == dats[0]

    def test_mixed_dat_repr(self, mdat):
        "MixedDat repr should produce a MixedDat object when eval'd."
        from pyop2.op2 import Set, DataSet, MixedDataSet, Dat, MixedDat  # noqa: needed by eval
        from numpy import dtype  # noqa: needed by eval
        assert isinstance(eval(repr(mdat)), op2.MixedDat)

    def test_mixed_dat_str(self, mdat):
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
        return op2.Sparsity((di, di), [(mi, mi, None)])

    @pytest.fixture
    def mixed_row_sparsity(cls, dtoset, mds, m_iterset_toset, mmap):
        return op2.Sparsity((mds, dtoset), {(0, 0): [(mmap[0], m_iterset_toset, None)],
                                            (1, 0): [(mmap[1], m_iterset_toset, None)]})

    @pytest.fixture
    def mixed_col_sparsity(cls, dtoset, mds, m_iterset_toset, mmap):
        return op2.Sparsity((dtoset, mds), {(0, 0): [(m_iterset_toset, mmap[0], None)],
                                            (0, 1): [(m_iterset_toset, mmap[1], None)]})

    def test_sparsity_illegal_rdset(self, di, mi):
        "Sparsity rdset should be a DataSet"
        with pytest.raises(TypeError):
            op2.Sparsity(('illegalrmap', di), [(mi, mi, None)])

    def test_sparsity_illegal_cdset(self, di, mi):
        "Sparsity cdset should be a DataSet"
        with pytest.raises(TypeError):
            op2.Sparsity((di, 'illegalrmap'), [(mi, mi, None)])

    def test_sparsity_illegal_rmap(self, di, mi):
        "Sparsity rmap should be a Map"
        with pytest.raises(TypeError):
            op2.Sparsity((di, di), [('illegalrmap', mi, None)])

    def test_sparsity_illegal_cmap(self, di, mi):
        "Sparsity cmap should be a Map"
        with pytest.raises(TypeError):
            op2.Sparsity((di, di), [(mi, 'illegalcmap', None)])

    def test_sparsity_illegal_name(self, di, mi):
        "Sparsity name should be a string."
        with pytest.raises(TypeError):
            op2.Sparsity((di, di), [(mi, mi, None)], 0)

    def test_sparsity_map_pair_different_dataset(self, mi, md, di, dd, m_iterset_toset):
        """Sparsity can be built from different row and column maps as long as
        the tosets match the row and column DataSet."""
        s = op2.Sparsity((di, dd), [(m_iterset_toset, md, None)], name="foo")
        assert (s.rcmaps[(0, 0)][0] == (m_iterset_toset, md) and s.dims[0][0] == (1, 1)
                and s.name == "foo" and s.dsets == (di, dd))

    def test_sparsity_unique_map_pairs(self, mi, di):
        "Sparsity constructor should filter duplicate tuples of pairs of maps."
        s = op2.Sparsity((di, di), [(mi, mi, None), (mi, mi, None)], name="foo")
        assert s.rcmaps[(0, 0)] == [(mi, mi)] and s.dims[0][0] == (1, 1)

    def test_sparsity_map_pairs_different_itset(self, mi, di, dd, m_iterset_toset):
        "Sparsity constructor should accept maps with different iteration sets"
        maps = ((m_iterset_toset, m_iterset_toset), (mi, mi))
        s = op2.Sparsity((di, di), [(*maps[0], None),
                                    (*maps[1], None)], name="foo")
        assert frozenset(s.rcmaps[(0, 0)]) == frozenset(maps) and s.dims[0][0] == (1, 1)

    def test_sparsity_map_pairs_sorted(self, mi, di, dd, m_iterset_toset):
        "Sparsity maps should have a deterministic order."
        s1 = op2.Sparsity((di, di), [(m_iterset_toset, m_iterset_toset, None), (mi, mi, None)])
        s2 = op2.Sparsity((di, di), [(mi, mi, None), (m_iterset_toset, m_iterset_toset, None)])
        assert s1.rcmaps[(0, 0)] == s2.rcmaps[(0, 0)]

    def test_sparsity_illegal_itersets(self, mi, md, di, dd):
        "Both maps in a (rmap,cmap) tuple must have same iteration set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((dd, di), [(md, mi, None)])

    def test_sparsity_illegal_row_datasets(self, mi, md, di):
        "All row maps must share the same data set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((di, di), [(mi, mi, None), (md, mi, None)])

    def test_sparsity_illegal_col_datasets(self, mi, md, di, dd):
        "All column maps must share the same data set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((di, di), [(mi, mi, None), (mi, md, None)])

    def test_sparsity_shape(self, s):
        "Sparsity shape of a single block should be (1, 1)."
        assert s.shape == (1, 1)

    def test_sparsity_iter(self, s):
        "Iterating over a Sparsity of a single block should yield self."
        for bs in s:
            assert bs == s

    def test_sparsity_getitem(self, s):
        "Block 0, 0 of a Sparsity of a single block should be self."
        assert s[0, 0] == s

    def test_sparsity_mmap_iter(self, ms):
        "Iterating a Sparsity should yield the block by row."
        cols = ms.shape[1]
        for i, block in enumerate(ms):
            assert block == ms[i // cols, i % cols]

    def test_sparsity_mmap_getitem(self, ms):
        """Sparsity block i, j should be defined on the corresponding row and
        column DataSets and Maps."""
        for i, rds in enumerate(ms.dsets[0]):
            for j, cds in enumerate(ms.dsets[1]):
                block = ms[i, j]
                # Indexing with a tuple and double index is equivalent
                assert block == ms[i][j]
                assert (block.dsets == (rds, cds)
                        and block.rcmaps[(0, 0)] == ms.rcmaps[(i, j)])

    def test_sparsity_mmap_getrow(self, ms):
        """Indexing a Sparsity with a single index should yield a row of
        blocks."""
        for i, rds in enumerate(ms.dsets[0]):
            for j, (s, cds) in enumerate(zip(ms[i], ms.dsets[1])):
                assert (s.dsets == (rds, cds)
                        and s.rcmaps[(0, 0)] == ms.rcmaps[(i, j)])

    def test_sparsity_mmap_shape(self, ms):
        "Sparsity shape of should be the sizes of the mixed space."
        assert ms.shape == (len(ms.dsets[0]), len(ms.dsets[1]))

    def test_sparsity_mmap_illegal_itersets(self, m_iterset_toset,
                                            m_iterset_set, m_set_toset,
                                            m_set_set, mds):
        "Both maps in a (rmap,cmap) tuple must have same iteration set."
        rmm = op2.MixedMap((m_iterset_toset, m_iterset_set))
        cmm = op2.MixedMap((m_set_toset, m_set_set))
        with pytest.raises(RuntimeError):
            op2.Sparsity((mds, mds), {(i, j): [(rm, cm, None)] for i, rm in enumerate(rmm) for j, cm in enumerate(cmm)})

    def test_sparsity_mmap_illegal_row_datasets(self, m_iterset_toset,
                                                m_iterset_set, m_set_toset, mds):
        "All row maps must share the same data set."
        rmm = op2.MixedMap((m_iterset_toset, m_iterset_set))
        cmm = op2.MixedMap((m_set_toset, m_set_toset))
        with pytest.raises(RuntimeError):
            op2.Sparsity((mds, mds), {(i, j): [(rm, cm, None)] for i, rm in enumerate(rmm) for j, cm in enumerate(cmm)})

    def test_sparsity_mmap_illegal_col_datasets(self, m_iterset_toset,
                                                m_iterset_set, m_set_toset, mds):
        "All column maps must share the same data set."
        rmm = op2.MixedMap((m_set_toset, m_set_toset))
        cmm = op2.MixedMap((m_iterset_toset, m_iterset_set))
        with pytest.raises(RuntimeError):
            op2.Sparsity((mds, mds), {(i, j): [(rm, cm, None)] for i, rm in enumerate(rmm) for j, cm in enumerate(cmm)})

    def test_sparsity_repr(self, sparsity):
        "Sparsity should have the expected repr."

        # Note: We can't actually reproduce a Sparsity from its repr because
        # the Sparsity constructor checks that the maps are populated
        r = "Sparsity(%r, %r, name=%r, nested=%r, block_sparse=%r, diagonal_block=%r)" % (sparsity.dsets, sparsity._maps_and_regions, sparsity.name, sparsity._nested, sparsity._block_sparse, sparsity._diagonal_block)
        assert repr(sparsity) == r

    def test_sparsity_str(self, sparsity):
        "Sparsity should have the expected string representation."
        s = "OP2 Sparsity: dsets %s, maps_and_regions %s, name %s, nested %s, block_sparse %s, diagonal_block %s" % \
            (sparsity.dsets, sparsity._maps_and_regions, sparsity.name, sparsity._nested, sparsity._block_sparse, sparsity._diagonal_block)
        assert str(sparsity) == s


class TestMatAPI:

    """
    Mat API unit tests
    """

    def test_mat_illegal_sets(self):
        "Mat sparsity should be a Sparsity."
        with pytest.raises(TypeError):
            op2.Mat('illegalsparsity')

    def test_mat_illegal_name(self, sparsity):
        "Mat name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Mat(sparsity, name=2)

    def test_mat_dtype(self, mat):
        "Default data type should be numpy.float64."
        assert mat.dtype == np.double

    def test_mat_properties(self, sparsity):
        "Mat constructor should correctly set attributes."
        m = op2.Mat(sparsity, 'double', 'bar')
        assert m.sparsity == sparsity and  \
            m.dtype == np.float64 and m.name == 'bar'

    def test_mat_mixed(self, mmat):
        "Default data type should be numpy.float64."
        assert mmat.dtype == np.double

    def test_mat_illegal_maps(self, mat):
        "Mat arg constructor should reject invalid maps."
        wrongmap = op2.Map(op2.Set(2), op2.Set(3), 2, [0, 0, 0, 0])
        with pytest.raises(exceptions.MapValueError):
            mat(op2.INC, (wrongmap, wrongmap))

    @pytest.mark.parametrize("mode", [op2.READ, op2.RW, op2.MIN, op2.MAX])
    def test_mat_arg_illegal_mode(self, mat, mode, m_iterset_toset):
        """Mat arg constructor should reject illegal access modes."""
        with pytest.raises(exceptions.ModeValueError):
            mat(mode, (m_iterset_toset, m_iterset_toset))

    def test_mat_iter(self, mat):
        "Mat should be iterable and yield self."
        for m in mat:
            assert m is mat

    def test_mat_repr(self, mat):
        "Mat should have the expected repr."

        # Note: We can't actually reproduce a Sparsity from its repr because
        # the Sparsity constructor checks that the maps are populated
        r = "Mat(%r, %r, %r)" % (mat.sparsity, mat.dtype, mat.name)
        assert repr(mat) == r

    def test_mat_str(self, mat):
        "Mat should have the expected string representation."
        s = "OP2 Mat: %s, sparsity (%s), datatype %s" \
            % (mat.name, mat.sparsity, mat.dtype.name)
        assert str(mat) == s


class TestGlobalAPI:

    """
    Global API unit tests
    """

    def test_global_illegal_dim(self):
        "Global dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Global('illegaldim', comm=COMM_WORLD)

    def test_global_illegal_dim_tuple(self):
        "Global dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Global((1, 'illegaldim'), comm=COMM_WORLD)

    def test_global_illegal_name(self):
        "Global name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Global(1, 1, name=2, comm=COMM_WORLD)

    def test_global_dim(self):
        "Global constructor should create a dim tuple."
        g = op2.Global(1, 1, comm=COMM_WORLD)
        assert g.dim == (1,)

    def test_global_dim_list(self):
        "Global constructor should create a dim tuple from a list."
        g = op2.Global([2, 3], [1] * 6, comm=COMM_WORLD)
        assert g.dim == (2, 3)

    def test_global_float(self):
        "Data type for float data should be numpy.float64."
        g = op2.Global(1, 1.0, comm=COMM_WORLD)
        assert g.dtype == np.asarray(1.0).dtype

    def test_global_int(self):
        "Data type for int data should be numpy.int."
        g = op2.Global(1, 1, comm=COMM_WORLD)
        assert g.dtype == np.asarray(1).dtype

    def test_global_convert_int_float(self):
        "Explicit float type should override NumPy's default choice of int."
        g = op2.Global(1, 1, dtype=np.float64, comm=COMM_WORLD)
        assert g.dtype == np.float64

    def test_global_convert_float_int(self):
        "Explicit int type should override NumPy's default choice of float."
        g = op2.Global(1, 1.5, dtype=np.int64, comm=COMM_WORLD)
        assert g.dtype == np.int64

    def test_global_illegal_dtype(self):
        "Illegal data type should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Global(1, 'illegal_type', 'double', comm=COMM_WORLD)

    @pytest.mark.parametrize("dim", [1, (2, 2)])
    def test_global_illegal_length(self, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Global(dim, [1] * (np.prod(dim) + 1), comm=COMM_WORLD)

    def test_global_reshape(self):
        "Data should be reshaped according to dim."
        g = op2.Global((2, 2), [1.0] * 4, comm=COMM_WORLD)
        assert g.dim == (2, 2) and g.data.shape == (2, 2)

    def test_global_properties(self):
        "Data globalructor should correctly set attributes."
        g = op2.Global((2, 2), [1] * 4, 'double', 'bar', comm=COMM_WORLD)
        assert g.dim == (2, 2) and g.dtype == np.float64 and g.name == 'bar' \
            and g.data.sum() == 4

    def test_global_setter(self, g):
        "Setter attribute on data should correct set data value."
        g.data = 2
        assert g.data.sum() == 2

    def test_global_setter_malformed_data(self, g):
        "Setter attribute should reject malformed data."
        with pytest.raises(exceptions.DataValueError):
            g.data = [1, 2]

    def test_global_iter(self, g):
        "Global should be iterable and yield self."
        for g_ in g:
            assert g_ is g

    def test_global_len(self, g):
        "Global len should be 1."
        assert len(g) == 1

    def test_global_str(self):
        "Global should have the expected string representation."
        g = op2.Global(1, 1, 'double', comm=COMM_WORLD)
        s = "OP2 Global Argument: %s with dim %s and value %s" \
            % (g.name, g.dim, g.data)
        assert str(g) == s

    @pytest.mark.parametrize("mode", [op2.RW, op2.WRITE])
    def test_global_arg_illegal_mode(self, g, mode):
        """Global __call__ should not allow illegal access modes."""
        with pytest.raises(exceptions.ModeValueError):
            g(mode)


class TestMapAPI:

    """
    Map API unit tests
    """

    def test_map_illegal_iterset(self, set):
        "Map iterset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Map('illegalset', set, 1, [])

    def test_map_illegal_toset(self, set):
        "Map toset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Map(set, 'illegalset', 1, [])

    def test_map_illegal_arity(self, set):
        "Map arity should be int."
        with pytest.raises(exceptions.ArityTypeError):
            op2.Map(set, set, 'illegalarity', [])

    def test_map_illegal_arity_tuple(self, set):
        "Map arity should not be a tuple."
        with pytest.raises(exceptions.ArityTypeError):
            op2.Map(set, set, (2, 2), [])

    def test_map_illegal_name(self, set):
        "Map name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Map(set, set, 1, [], name=2)

    def test_map_illegal_dtype(self, set):
        "Illegal data type should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Map(set, set, 1, 'abcdefg')

    def test_map_illegal_length(self, iterset, toset):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Map(iterset, toset, 1, [1] * (iterset.size + 1))

    def test_map_convert_float_int(self, iterset, toset):
        "Float data should be implicitely converted to int."
        from pyop2.datatypes import IntType
        m = op2.Map(iterset, toset, 1, [1.5] * iterset.size)
        assert m.values.dtype == IntType and m.values.sum() == iterset.size

    def test_map_reshape(self, iterset, toset):
        "Data should be reshaped according to arity."
        m = op2.Map(iterset, toset, 2, [1] * 2 * iterset.size)
        assert m.arity == 2 and m.values.shape == (iterset.size, 2)

    def test_map_split(self, m_iterset_toset):
        "Splitting a Map should yield a tuple with self"
        for m in m_iterset_toset.split:
            m == m_iterset_toset

    def test_map_properties(self, iterset, toset):
        "Data constructor should correctly set attributes."
        m = op2.Map(iterset, toset, 2, [1] * 2 * iterset.size, 'bar')
        assert (m.iterset == iterset and m.toset == toset and m.arity == 2
                and m.arities == (2,) and m.arange == (0, 2)
                and m.values.sum() == 2 * iterset.size and m.name == 'bar')

    def test_map_eq(self, m_iterset_toset):
        """Map equality is identity."""
        mcopy = op2.Map(m_iterset_toset.iterset, m_iterset_toset.toset,
                        m_iterset_toset.arity, m_iterset_toset.values)
        assert m_iterset_toset != mcopy
        assert not m_iterset_toset == mcopy
        assert mcopy == mcopy

    def test_map_ne_iterset(self, m_iterset_toset):
        """Maps that have copied but not equal iteration sets are not equal."""
        mcopy = op2.Map(op2.Set(m_iterset_toset.iterset.size),
                        m_iterset_toset.toset, m_iterset_toset.arity,
                        m_iterset_toset.values)
        assert m_iterset_toset != mcopy
        assert not m_iterset_toset == mcopy

    def test_map_ne_toset(self, m_iterset_toset):
        """Maps that have copied but not equal to sets are not equal."""
        mcopy = op2.Map(m_iterset_toset.iterset, op2.Set(m_iterset_toset.toset.size),
                        m_iterset_toset.arity, m_iterset_toset.values)
        assert m_iterset_toset != mcopy
        assert not m_iterset_toset == mcopy

    def test_map_ne_arity(self, m_iterset_toset):
        """Maps that have different arities are not equal."""
        mcopy = op2.Map(m_iterset_toset.iterset, m_iterset_toset.toset,
                        m_iterset_toset.arity * 2, list(m_iterset_toset.values) * 2)
        assert m_iterset_toset != mcopy
        assert not m_iterset_toset == mcopy

    def test_map_ne_values(self, m_iterset_toset):
        """Maps that have different values are not equal."""
        m2 = op2.Map(m_iterset_toset.iterset, m_iterset_toset.toset,
                     m_iterset_toset.arity, m_iterset_toset.values.copy())
        m2.values[0] = 2
        assert m_iterset_toset != m2
        assert not m_iterset_toset == m2

    def test_map_iter(self, m_iterset_toset):
        "Map should be iterable and yield self."
        for m_ in m_iterset_toset:
            assert m_ is m_iterset_toset

    def test_map_len(self, m_iterset_toset):
        "Map len should be 1."
        assert len(m_iterset_toset) == 1

    def test_map_repr(self, m_iterset_toset):
        "Map should have the expected repr."
        r = "Map(%r, %r, %r, None, %r, %r, %r)" % (m_iterset_toset.iterset, m_iterset_toset.toset,
                                                   m_iterset_toset.arity, m_iterset_toset.name, m_iterset_toset._offset, m_iterset_toset._offset_quotient)
        assert repr(m_iterset_toset) == r

    def test_map_str(self, m_iterset_toset):
        "Map should have the expected string representation."
        s = "OP2 Map: %s from (%s) to (%s) with arity %s" \
            % (m_iterset_toset.name, m_iterset_toset.iterset, m_iterset_toset.toset, m_iterset_toset.arity)
        assert str(m_iterset_toset) == s


class TestMixedMapAPI:

    """
    MixedMap API unit tests
    """

    def test_mixed_map_illegal_arg(self):
        "Map iterset should be Set."
        with pytest.raises(TypeError):
            op2.MixedMap('illegalarg')

    def test_mixed_map_split(self, maps):
        """Constructing a MixedDat from an iterable of Maps should leave them
        unchanged."""
        mmap = op2.MixedMap(maps)
        assert mmap.split == maps
        for i, m in enumerate(maps):
            assert mmap.split[i] == m
        assert mmap.split[:-1] == tuple(mmap)[:-1]

    def test_mixed_map_iterset(self, mmap):
        "MixedMap iterset should return the common iterset of all Maps."
        for m in mmap:
            assert mmap.iterset == m.iterset

    def test_mixed_map_toset(self, mmap):
        "MixedMap toset should return a MixedSet of the Map tosets."
        assert mmap.toset == op2.MixedSet(m.toset for m in mmap)

    def test_mixed_map_arity(self, mmap):
        "MixedMap arity should return the sum of the Map arities."
        assert mmap.arity == sum(m.arity for m in mmap)

    def test_mixed_map_arities(self, mmap):
        "MixedMap arities should return a tuple of the Map arities."
        assert mmap.arities == tuple(m.arity for m in mmap)

    def test_mixed_map_arange(self, mmap):
        "MixedMap arities should return a tuple of the Map arities."
        assert mmap.arange == (0,) + tuple(np.cumsum(mmap.arities))

    def test_mixed_map_values(self, mmap):
        "MixedMap values should return a tuple of the Map values."
        assert all((v == m.values).all() for v, m in zip(mmap.values, mmap))

    def test_mixed_map_values_with_halo(self, mmap):
        "MixedMap values_with_halo should return a tuple of the Map values."
        assert all((v == m.values_with_halo).all() for v, m in zip(mmap.values_with_halo, mmap))

    def test_mixed_map_name(self, mmap):
        "MixedMap name should return a tuple of the Map names."
        assert mmap.name == tuple(m.name for m in mmap)

    def test_mixed_map_offset(self, mmap):
        "MixedMap offset should return a tuple of the Map offsets."
        assert mmap.offset == tuple(m.offset for m in mmap)

    def test_mixed_map_iter(self, maps):
        "MixedMap should be iterable and yield the Maps."
        assert tuple(m for m in op2.MixedMap(maps)) == maps

    def test_mixed_map_len(self, maps):
        """MixedMap should have length equal to the number of contained Maps."""
        assert len(op2.MixedMap(maps)) == len(maps)

    def test_mixed_map_eq(self, maps):
        "MixedMaps created from the same Maps should compare equal."
        assert op2.MixedMap(maps) == op2.MixedMap(maps)
        assert not op2.MixedMap(maps) != op2.MixedMap(maps)

    def test_mixed_map_ne(self, maps):
        "MixedMaps created from different Maps should not compare equal."
        mm1 = op2.MixedMap((maps[0], maps[1]))
        mm2 = op2.MixedMap((maps[1], maps[0]))
        assert mm1 != mm2
        assert not mm1 == mm2

    def test_mixed_map_ne_map(self, maps):
        "A MixedMap should not compare equal to a Map."
        assert op2.MixedMap(maps) != maps[0]
        assert not op2.MixedMap(maps) == maps[0]

    def test_mixed_map_repr(self, mmap):
        "MixedMap should have the expected repr."
        # Note: We can't actually reproduce a MixedMap from its repr because
        # the iteration sets will not be identical, which is checked in the
        # constructor
        assert repr(mmap) == "MixedMap(%r)" % (mmap.split,)

    def test_mixed_map_str(self, mmap):
        "MixedMap should have the expected string representation."
        assert str(mmap) == "OP2 MixedMap composed of Maps: %s" % (mmap.split,)


class TestKernelAPI:

    """
    Kernel API unit tests
    """

    def test_kernel_illegal_name(self):
        "Kernel name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Kernel("", name=2)

    def test_kernel_properties(self):
        "Kernel constructor should correctly set attributes."
        k = op2.CStringLocalKernel("", "foo", accesses=(), dtypes=())
        assert k.name == "foo"

    def test_kernel_repr(self, set):
        "Kernel should have the expected repr."
        k = op2.Kernel("static int foo() { return 0; }", 'foo')
        assert repr(k) == 'Kernel("""%s""", %r)' % (k.code, k.name)

    def test_kernel_str(self, set):
        "Kernel should have the expected string representation."
        k = op2.Kernel("static int foo() { return 0; }", 'foo')
        assert str(k) == "OP2 Kernel: %s" % k.name


class TestParLoopAPI:

    """
    ParLoop API unit tests
    """

    def test_illegal_kernel(self, set, dat, m_iterset_toset):
        """The first ParLoop argument has to be of type op2.Kernel."""
        with pytest.raises(exceptions.KernelTypeError):
            op2.par_loop('illegal_kernel', set, dat(op2.READ, m_iterset_toset))

    def test_illegal_iterset(self, dat, m_iterset_toset):
        """The first ParLoop argument has to be of type op2.Kernel."""
        with pytest.raises(exceptions.SetTypeError):
            op2.par_loop(op2.Kernel("", "k"), 'illegal_set',
                         dat(op2.READ, m_iterset_toset))

    def test_illegal_dat_iterset(self):
        """ParLoop should reject a Dat argument using a different iteration
        set from the par_loop's."""
        set1 = op2.Set(2)
        set2 = op2.Set(3)
        dset1 = op2.DataSet(set1, 1)
        dat = op2.Dat(dset1)
        map = op2.Map(set2, set1, 1, [0, 0, 0])
        kernel = op2.Kernel("void k() { }", "k")
        with pytest.raises(exceptions.MapValueError):
            op2.ParLoop(kernel, set1, dat(op2.READ, map))

    def test_illegal_mat_iterset(self, sparsity):
        """ParLoop should reject a Mat argument using a different iteration
        set from the par_loop's."""
        set1 = op2.Set(2)
        m = op2.Mat(sparsity)
        rmap, cmap = sparsity.rcmaps[(0, 0)][0]
        kernel = op2.Kernel("static void k() { }", "k")
        with pytest.raises(exceptions.MapValueError):
            op2.par_loop(
                kernel,
                set1,
                m(op2.INC, (rmap, cmap))
            )

    def test_empty_map_and_iterset(self):
        """If the iterset of the ParLoop is zero-sized, it should not matter if
        a map defined on it has no values."""
        s1 = op2.Set(0)
        s2 = op2.Set(10)
        m = op2.Map(s1, s2, 3)
        d = op2.Dat(s2 ** 1, [0] * 10, dtype=int)
        k = op2.Kernel("static void k(int *x) {}", "k")
        op2.par_loop(k, s1, d(op2.READ, m))

    def test_frozen_dats_cannot_use_different_access_mode(self):
        s1 = op2.Set(2)
        s2 = op2.Set(3)
        m = op2.Map(s1, s2, 3, [0]*6)
        d = op2.Dat(s2**1, [0]*3, dtype=int)
        k = op2.Kernel("static void k(int *x) {}", "k")

        with d.frozen_halo(op2.INC):
            op2.par_loop(k, s1, d(op2.INC, m))

            with pytest.raises(RuntimeError):
                op2.par_loop(k, s1, d(op2.WRITE, m))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
