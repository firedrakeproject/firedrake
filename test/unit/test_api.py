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

from pyop2 import op2
from pyop2 import exceptions
from pyop2 import sequential
from pyop2 import base
from pyop2 import configuration as cfg

@pytest.fixture
def set():
    return op2.Set(5, 'foo')

@pytest.fixture
def iterset():
    return op2.Set(2, 'iterset')

@pytest.fixture
def dataset():
    return op2.Set(3, 'dataset')

@pytest.fixture
def m(iterset, dataset):
    return op2.Map(iterset,  dataset, 2, [1] * 2 * iterset.size, 'm')

@pytest.fixture
def const(request):
    c = op2.Const(1, 1, 'test_const_nonunique_name')
    request.addfinalizer(c.remove_from_namespace)
    return c

@pytest.fixture
def sparsity(m):
    return op2.Sparsity((m, m), 1)

class TestInitAPI:
    """
    Init API unit tests
    """

    def test_noninit(self):
        "RuntimeError should be raised when using op2 before calling init."
        with pytest.raises(RuntimeError):
            op2.Set(1)

    def test_invalid_init(self):
        "init should not accept an invalid backend."
        with pytest.raises(ImportError):
            op2.init(backend='invalid_backend')

    def test_init(self, backend):
        "init should correctly set the backend."
        assert op2.backends.get_backend() == 'pyop2.'+backend

    def test_double_init(self, backend):
        "Calling init again with the same backend should update the configuration."
        op2.init(backend=backend, foo='bar')
        assert op2.backends.get_backend() == 'pyop2.'+backend
        assert cfg.foo == 'bar'

    def test_change_backend_fails(self, backend):
        "Calling init again with a different backend should fail."
        with pytest.raises(RuntimeError):
            op2.init(backend='other')

class TestAccessAPI:
    """
    Access API unit tests
    """

    @pytest.mark.parametrize("mode", base.Access._modes)
    def test_access(self, backend, mode):
        "Access repr should have the expected format."
        a = base.Access(mode)
        assert repr(a) == "Access('%s')" % mode

    def test_illegal_access(self, backend):
        "Illegal access modes should raise an exception."
        with pytest.raises(exceptions.ModeValueError):
            base.Access('ILLEGAL_ACCESS')

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
            op2.Set(1,2)

    def test_set_properties(self, backend, set):
        "Set constructor should correctly initialise attributes."
        assert set.size == 5 and set.name == 'foo'

    def test_set_repr(self, backend, set):
        "Set repr should have the expected format."
        assert repr(set) == "Set(5, 'foo')"

    def test_set_str(self, backend, set):
        "Set string representation should have the expected format."
        assert str(set) == "OP2 Set: foo with size 5"

    def test_set_equality(self, backend, set):
        "The equality test for sets is identity, not attribute equality"
        setcopy = op2.Set(set.size, set.name)
        assert set == set and set != setcopy

    # FIXME: test Set._lib_handle

class TestDatAPI:
    """
    Dat API unit tests
    """

    def test_dat_illegal_set(self, backend):
        "Dat set should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Dat('illegalset', 1)

    def test_dat_illegal_dim(self, backend, set):
        "Dat dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Dat(set, 'illegaldim')

    def test_dat_illegal_dim_tuple(self, backend, set):
        "Dat dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Dat(set, (1,'illegaldim'))

    def test_dat_illegal_name(self, backend, set):
        "Dat name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Dat(set, 1, name=2)

    def test_dat_initialise_data(self, backend, set):
        """Dat initilialised without the data should initialise data with the
        correct size and type."""
        d = op2.Dat(set, 1)
        assert d.data.size == 5 and d.data.dtype == np.float64

    def test_dat_initialise_vector_data(self, backend, set):
        """Dat initilialised without the data should initialise data with the
        correct size and type - vector data case."""
        d = op2.Dat(set, 2)
        assert d.data.size == 10 and d.data.dtype == np.float64

    def test_dat_initialise_dimlist_data(self, backend, set):
        """Dat initilialised without the data should initialise data with the
        correct size and type - list of dims case."""
        d = op2.Dat(set, [2, 3])
        assert d.data.size == 30 and d.data.dtype == np.float64

    def test_dat_initialise_data_type(self, backend, set):
        """Dat intiialised without the data but with specified type should
        initialise its data with the correct type."""
        d = op2.Dat(set, 1, dtype=np.int32)
        assert d.data.size == 5 and d.data.dtype == np.int32

    def test_dat_illegal_map(self, backend, set):
        """Dat __call__ should not allow a map with a dataset other than this
        Dat's set."""
        d = op2.Dat(set, 1)
        set1 = op2.Set(3)
        set2 = op2.Set(2)
        to_set2 = op2.Map(set1, set2, 1, [0, 0, 0])
        with pytest.raises(exceptions.MapValueError):
            d(to_set2, op2.READ)

    def test_dat_dim(self, backend, set):
        "Dat constructor should create a dim tuple."
        d = op2.Dat(set, 1)
        assert d.dim == (1,)

    def test_dat_dim_list(self, backend, set):
        "Dat constructor should create a dim tuple from a list."
        d = op2.Dat(set, [2,3])
        assert d.dim == (2,3)

    def test_dat_dtype(self, backend, set):
        "Default data type should be numpy.float64."
        d = op2.Dat(set, 1)
        assert d.dtype == np.double

    def test_dat_float(self, backend, set):
        "Data type for float data should be numpy.float64."
        d = op2.Dat(set, 1, [1.0]*set.size)
        assert d.dtype == np.double

    def test_dat_int(self, backend, set):
        "Data type for int data should be numpy.int."
        d = op2.Dat(set, 1, [1]*set.size)
        assert d.dtype == np.int

    def test_dat_convert_int_float(self, backend, set):
        "Explicit float type should override NumPy's default choice of int."
        d = op2.Dat(set, 1, [1]*set.size, np.double)
        assert d.dtype == np.float64

    def test_dat_convert_float_int(self, backend, set):
        "Explicit int type should override NumPy's default choice of float."
        d = op2.Dat(set, 1, [1.5]*set.size, np.int32)
        assert d.dtype == np.int32

    def test_dat_illegal_dtype(self, backend, set):
        "Illegal data type should raise DataTypeError."
        with pytest.raises(exceptions.DataTypeError):
            op2.Dat(set, 1, dtype='illegal_type')

    @pytest.mark.parametrize("dim", [1, (2,2)])
    def test_dat_illegal_length(self, backend, set, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Dat(set, dim, [1]*(set.size*np.prod(dim)+1))

    def test_dat_reshape(self, backend, set):
        "Data should be reshaped according to dim."
        d = op2.Dat(set, (2,2), [1.0]*set.size*4)
        assert d.dim == (2,2) and d.data.shape == (set.size,2,2)

    def test_dat_properties(self, backend, set):
        "Dat constructor should correctly set attributes."
        d = op2.Dat(set, (2,2), [1]*set.size*4, 'double', 'bar')
        assert d.dataset == set and d.dim == (2,2) and \
                d.dtype == np.float64 and d.name == 'bar' and \
                d.data.sum() == set.size*4

    def test_dat_ro_accessor(self, backend, set):
        "Attempting to set values through the RO accessor should raise an error."
        d = op2.Dat(set, 2, range(2 * set.size), dtype=np.int32)
        x = d.data_ro
        with pytest.raises((RuntimeError, ValueError)):
            x[0] = 1

    def test_dat_ro_write_accessor(self, backend, set):
        "Re-accessing the data in writeable form should be allowed."
        d = op2.Dat(set, 1, range(set.size), dtype=np.int32)
        x = d.data_ro
        with pytest.raises((RuntimeError, ValueError)):
            x[0] = 1
        x = d.data
        x[0] = -100
        assert d.data_ro[0] == -100

class TestSparsityAPI:
    """
    Sparsity API unit tests
    """

    @pytest.fixture
    def mi(cls, dataset):
        iterset = op2.Set(3, 'iterset2')
        return op2.Map(iterset,  dataset, 1, [1] * iterset.size, 'mi')

    @pytest.fixture
    def md(cls, iterset):
        dataset = op2.Set(1, 'dataset2')
        return op2.Map(iterset,  dataset, 1, [1] * iterset.size, 'md')

    def test_sparsity_illegal_rmap(self, backend, m):
        "Sparsity rmap should be a Map"
        with pytest.raises(TypeError):
            op2.Sparsity(('illegalrmap', m), 1)

    def test_sparsity_illegal_cmap(self, backend, m):
        "Sparsity cmap should be a Map"
        with pytest.raises(TypeError):
            op2.Sparsity((m, 'illegalcmap'), 1)

    def test_sparsity_illegal_dim(self, backend, m):
        "Sparsity dim should be an int"
        with pytest.raises(TypeError):
            op2.Sparsity((m, m), 'illegaldim')

    def test_sparsity_single_map(self, backend, m):
        "Sparsity constructor should accept single Map and turn it into tuple"
        s = op2.Sparsity(m, 2, "foo")
        assert s.maps[0] == (m, m) and s.dims == (2,2) and s.name == "foo"

    def test_sparsity_map_pair(self, backend, m):
        "Sparsity constructor should accept a pair of maps"
        s = op2.Sparsity((m, m), 2, "foo")
        assert s.maps[0] == (m, m) and s.dims == (2,2) and s.name == "foo"

    def test_sparsity_map_pair_different_dataset(self, backend, m, md):
        "Sparsity constructor should accept a pair of maps"
        s = op2.Sparsity((m, md), 2, "foo")
        assert s.maps[0] == (m, md) and s.dims == (2,2) and s.name == "foo"

    def test_sparsity_multiple_map_pairs(self, backend, m):
        "Sparsity constructor should accept tuple of pairs of maps"
        s = op2.Sparsity(((m, m), (m, m)),
                         1, "foo")
        assert s.maps == [(m, m), (m, m)] and s.dims == (1,1)

    def test_sparsity_map_pairs_different_itset(self, backend, m, mi):
        "Sparsity constructor should accept maps with different iteration sets"
        s = op2.Sparsity(((m, m), (mi, mi)),
                         1, "foo")
        assert s.maps == [(m, m), (mi, mi)] and s.dims == (1,1)

    def test_sparsity_illegal_itersets(self, backend, m, mi):
        "Both maps in a (rmap,cmap) tuple must have same iteration set"
        with pytest.raises(RuntimeError):
            op2.Sparsity((m, mi), 1)

    def test_sparsity_illegal_row_datasets(self, backend, m, md):
        "All row maps must share the same data set"
        with pytest.raises(RuntimeError):
            op2.Sparsity(((m, m), (md, m)), 1)

    def test_sparsity_illegal_col_datasets(self, backend, m, md):
        "All column maps must share the same data set"
        with pytest.raises(RuntimeError):
            op2.Sparsity(((m, m), (m, md)), 1)

class TestMatAPI:
    """
    Mat API unit tests
    """

    def test_mat_illegal_sets(self, backend):
        "Mat sparsity should be a Sparsity."
        with pytest.raises(TypeError):
            op2.Mat('illegalsparsity')

    def test_mat_illegal_name(self, backend, sparsity):
        "Mat name should be string."
        with pytest.raises(sequential.NameTypeError):
            op2.Mat(sparsity, name=2)

    def test_mat_dtype(self, backend, sparsity):
        "Default data type should be numpy.float64."
        m = op2.Mat(sparsity)
        assert m.dtype == np.double

    def test_mat_properties(self, backend, sparsity):
        "Mat constructor should correctly set attributes."
        m = op2.Mat(sparsity, 'double', 'bar')
        assert m.sparsity == sparsity and  \
                m.dtype == np.float64 and m.name == 'bar'

    def test_mat_illegal_maps(self, backend, sparsity):
        m = op2.Mat(sparsity)
        set1 = op2.Set(2)
        set2 = op2.Set(3)
        wrongmap = op2.Map(set1, set2, 2, [0, 0, 0, 0])
        with pytest.raises(exceptions.MapValueError):
            m((wrongmap[0], wrongmap[1]), op2.INC)


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
            op2.Const((1,'illegaldim'), 1, 'test_const_illegal_dim_tuple')

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
        c = op2.Const([2,3], [1]*6, 'test_const_dim_list')
        c.remove_from_namespace()
        assert c.dim == (2,3)

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

    @pytest.mark.parametrize("dim", [1, (2,2)])
    def test_const_illegal_length(self, backend, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Const(dim, [1]*(np.prod(dim)+1), 'test_const_illegal_length_%r' % np.prod(dim))

    def test_const_reshape(self, backend):
        "Data should be reshaped according to dim."
        c = op2.Const((2,2), [1.0]*4, 'test_const_reshape')
        c.remove_from_namespace()
        assert c.dim == (2,2) and c.data.shape == (2,2)

    def test_const_properties(self, backend):
        "Data constructor should correctly set attributes."
        c = op2.Const((2,2), [1]*4, 'baz', 'double')
        c.remove_from_namespace()
        assert c.dim == (2,2) and c.dtype == np.float64 and c.name == 'baz' \
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
            op2.Global((1,'illegaldim'))

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
        g = op2.Global([2,3], [1]*6)
        assert g.dim == (2,3)

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

    @pytest.mark.parametrize("dim", [1, (2,2)])
    def test_global_illegal_length(self, backend, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Global(dim, [1]*(np.prod(dim)+1))

    def test_global_reshape(self, backend):
        "Data should be reshaped according to dim."
        g = op2.Global((2,2), [1.0]*4)
        assert g.dim == (2,2) and g.data.shape == (2,2)

    def test_global_properties(self, backend):
        "Data globalructor should correctly set attributes."
        g = op2.Global((2,2), [1]*4, 'double', 'bar')
        assert g.dim == (2,2) and g.dtype == np.float64 and g.name == 'bar' \
                and g.data.sum() == 4

    def test_global_setter(self, backend):
        "Setter attribute on data should correct set data value."
        c = op2.Global(1, 1)
        c.data = 2
        assert c.data.sum() == 2

    def test_global_setter_malformed_data(self, backend):
        "Setter attribute should reject malformed data."
        c = op2.Global(1, 1)
        with pytest.raises(exceptions.DataValueError):
            c.data = [1, 2]

class TestMapAPI:
    """
    Map API unit tests
    """

    def test_map_illegal_iterset(self, backend, set):
        "Map iterset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Map('illegalset', set, 1, [])

    def test_map_illegal_dataset(self, backend, set):
        "Map dataset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.Map(set, 'illegalset', 1, [])

    def test_map_illegal_dim(self, backend, set):
        "Map dim should be int."
        with pytest.raises(exceptions.DimTypeError):
            op2.Map(set, set, 'illegaldim', [])

    def test_map_illegal_dim_tuple(self, backend, set):
        "Map dim should not be a tuple."
        with pytest.raises(exceptions.DimTypeError):
            op2.Map(set, set, (2,2), [])

    def test_map_illegal_name(self, backend, set):
        "Map name should be string."
        with pytest.raises(exceptions.NameTypeError):
            op2.Map(set, set, 1, [], name=2)

    def test_map_illegal_dtype(self, backend, set):
        "Illegal data type should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Map(set, set, 1, 'abcdefg')

    def test_map_illegal_length(self, backend, iterset, dataset):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(exceptions.DataValueError):
            op2.Map(iterset, dataset, 1, [1]*(iterset.size+1))

    def test_map_convert_float_int(self, backend, iterset, dataset):
        "Float data should be implicitely converted to int."
        m = op2.Map(iterset, dataset, 1, [1.5]*iterset.size)
        assert m.values.dtype == np.int32 and m.values.sum() == iterset.size

    def test_map_reshape(self, backend, iterset, dataset):
        "Data should be reshaped according to dim."
        m = op2.Map(iterset, dataset, 2, [1]*2*iterset.size)
        assert m.dim == 2 and m.values.shape == (iterset.size,2)

    def test_map_properties(self, backend, iterset, dataset):
        "Data constructor should correctly set attributes."
        m = op2.Map(iterset, dataset, 2, [1]*2*iterset.size, 'bar')
        assert m.iterset == iterset and m.dataset == dataset and m.dim == 2 \
                and m.values.sum() == 2*iterset.size and m.name == 'bar'

    def test_map_indexing(self, backend, iterset, dataset):
        "Indexing a map should create an appropriate Arg"
        m = op2.Map(iterset, dataset, 2, [1] * 2 * iterset.size, 'm')

        arg = m[0]
        assert arg.idx == 0

    def test_map_slicing(self, backend, iterset, dataset):
        "Slicing a map is not allowed"
        m = op2.Map(iterset, dataset, 2, [1] * 2 * iterset.size, 'm')

        with pytest.raises(NotImplementedError):
            arg = m[:]

    def test_map_equality(self, backend, m):
        """A map is equal if all its attributes are equal, bearing in mind that
        equality is identity for sets."""
        m2 = op2.Map(m.iterset, m.dataset, m.dim, m.values, m.name)
        assert m == m2

    def test_map_copied_set_inequality(self, backend, m):
        """Maps that have copied but not equal iteration sets are not equal"""
        itercopy = op2.Set(m.iterset.size, m.iterset.name)
        m2 = op2.Map(itercopy, m.dataset, m.dim, m.values, m.name)
        assert m != m2

    def test_map_dimension_inequality(self, backend, m):
        """Maps that have different dimensions are not equal"""
        m2 = op2.Map(m.iterset, m.dataset, m.dim*2, list(m.values)*2, m.name)
        assert m != m2

    def test_map_name_inequality(self, backend, m):
        """Maps with different names are not equal"""
        n = op2.Map(m.iterset, m.dataset, m.dim, m.values, 'n')
        assert m != n

class TestIterationSpaceAPI:
    """
    IterationSpace API unit tests
    """

    def test_iteration_space_illegal_iterset(self, backend, set):
        "IterationSpace iterset should be Set."
        with pytest.raises(exceptions.SetTypeError):
            op2.IterationSpace('illegalset', 1)

    def test_iteration_space_illegal_extents(self, backend, set):
        "IterationSpace extents should be int or int tuple."
        with pytest.raises(TypeError):
            op2.IterationSpace(set, 'illegalextents')

    def test_iteration_space_illegal_extents_tuple(self, backend, set):
        "IterationSpace extents should be int or int tuple."
        with pytest.raises(TypeError):
            op2.IterationSpace(set, (1,'illegalextents'))

    def test_iteration_space_extents(self, backend, set):
        "IterationSpace constructor should create a extents tuple."
        m = op2.IterationSpace(set, 1)
        assert m.extents == (1,)

    def test_iteration_space_extents_list(self, backend, set):
        "IterationSpace constructor should create a extents tuple from a list."
        m = op2.IterationSpace(set, [2,3])
        assert m.extents == (2,3)

    def test_iteration_space_properties(self, backend, set):
        "IterationSpace constructor should correctly set attributes."
        i = op2.IterationSpace(set, (2,3))
        assert i.iterset == set and i.extents == (2,3)

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

class TestIllegalItersetMaps:
    """
    Pass args with the wrong iterset maps to ParLoops, and check that they are trapped.
    """

    def test_illegal_dat_iterset(self, backend):
        set1 = op2.Set(2)
        set2 = op2.Set(3)
        dat = op2.Dat(set1, 1)
        map = op2.Map(set2, set1, 1, [0, 0, 0])
        kernel = op2.Kernel("void k() { }", "k")
        with pytest.raises(exceptions.MapValueError):
            base.ParLoop(kernel, set1, dat(map, op2.READ))

    def test_illegal_mat_iterset(self, backend, sparsity):
        set1 = op2.Set(2)
        m = op2.Mat(sparsity)
        rmap, cmap = sparsity.maps[0]
        kernel = op2.Kernel("void k() { }", "k")
        with pytest.raises(exceptions.MapValueError):
            base.ParLoop(kernel, set1(3,3), m((rmap[op2.i[0]], cmap[op2.i[1]]), op2.INC))

class TestSolverAPI:
    """
    Test the Solver API.
    """

    def test_solver_defaults(self, backend):
        s = op2.Solver()
        assert s.parameters == base.DEFAULT_SOLVER_PARAMETERS

    def test_set_options_with_params(self, backend):
        params = {'linear_solver': 'gmres',
                  'maximum_iterations': 25 }
        s = op2.Solver(params)
        assert s.parameters['linear_solver'] == 'gmres' \
           and s.parameters['maximum_iterations'] == 25

    def test_set_options_with_kwargs(self, backend):
        s = op2.Solver(linear_solver='gmres', maximum_iterations=25)
        assert s.parameters['linear_solver'] == 'gmres' \
           and s.parameters['maximum_iterations'] == 25

    def test_update_parameters(self, backend):
        s = op2.Solver()
        params = {'linear_solver': 'gmres',
                  'maximum_iterations': 25 }
        s.update_parameters(params)
        assert s.parameters['linear_solver'] == 'gmres' \
           and s.parameters['maximum_iterations'] == 25

    def test_set_params_and_kwargs_illegal(self, backend):
        params = {'linear_solver': 'gmres',
                  'maximum_iterations': 25 }
        with pytest.raises(RuntimeError):
            op2.Solver(params, linear_solver='cgs')

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
