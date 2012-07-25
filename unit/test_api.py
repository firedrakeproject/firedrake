import pytest
import numpy as np
import h5py

from pyop2 import op2
from pyop2 import sequential

def pytest_funcarg__set(request):
    return op2.Set(5, 'foo')

def pytest_funcarg__iterset(request):
    return op2.Set(2, 'iterset')

def pytest_funcarg__dataset(request):
    return op2.Set(3, 'dataset')

def pytest_funcarg__const(request):
    return request.cached_setup(scope='function',
            setup=lambda: op2.Const(1, 1, 'test_const_nonunique_name'),
            teardown=lambda c: c.remove_from_namespace())

def pytest_funcarg__h5file(request):
    tmpdir = request.getfuncargvalue('tmpdir')
    def make_hdf5_file():
        f = h5py.File(str(tmpdir.join('tmp_hdf5.h5')), 'w')
        f.create_dataset('dat', data=np.arange(10).reshape(5,2),
                         dtype=np.float64)
        f['dat'].attrs['type'] = 'double'
        f.create_dataset('soadat', data=np.arange(10).reshape(2,5),
                         dtype=np.float64)
        f['soadat'].attrs['type'] = 'double:soa'
        f.create_dataset('set', data=np.array((5,)))
        f.create_dataset('constant', data=np.arange(3))
        f.create_dataset('map', data=np.array((1,2,2,3)).reshape(2,2))
        return f

    return request.cached_setup(scope='module',
                                setup=lambda: make_hdf5_file(),
                                teardown=lambda f: f.close())

class TestUserAPI:
    """
    User API Unit Tests
    """

    ## Init unit tests

    def test_noninit(self):
        "RuntimeError should be raised when using op2 before calling init."
        with pytest.raises(RuntimeError):
            op2.Set(1)

    def test_invalid_init(self):
        "init should only be callable once."
        with pytest.raises(ValueError):
            op2.init('invalid_backend')

    def test_init(self, backend):
        "init should correctly set the backend."
        assert op2.backends.get_backend() == 'pyop2.'+backend

    def test_double_init(self, backend):
        "init should only be callable once."
        with pytest.raises(RuntimeError):
            op2.init(backend)

    def test_init_exit(self, backend):
        op2.exit()
        op2.init(backend)

    ## Set unit tests

    def test_set_illegal_size(self, backend):
        "Set size should be int."
        with pytest.raises(sequential.SizeTypeError):
            op2.Set('illegalsize')

    def test_set_illegal_name(self, backend):
        "Set name should be string."
        with pytest.raises(sequential.NameTypeError):
            op2.Set(1,2)

    def test_set_properties(self, set, backend):
        "Set constructor should correctly initialise attributes."
        assert set.size == 5 and set.name == 'foo'

    def test_set_repr(self, set, backend):
        "Set repr should have the expected format."
        assert repr(set) == "Set(5, 'foo')"

    def test_set_str(self, set, backend):
        "Set string representation should have the expected format."
        assert str(set) == "OP2 Set: foo with size 5"

    def test_set_hdf5(self, h5file, backend):
        "Set should get correct size from HDF5 file."
        s = op2.Set.fromhdf5(h5file, name='set')
        assert s.size == 5
    # FIXME: test Set._lib_handle

    ## Dat unit tests

    def test_dat_illegal_set(self, backend):
        "Dat set should be Set."
        with pytest.raises(sequential.SetTypeError):
            op2.Dat('illegalset', 1)

    def test_dat_illegal_dim(self, set, backend):
        "Dat dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Dat(set, 'illegaldim')

    def test_dat_illegal_dim_tuple(self, set, backend):
        "Dat dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Dat(set, (1,'illegaldim'))

    def test_dat_illegal_name(self, set, backend):
        "Dat name should be string."
        with pytest.raises(sequential.NameTypeError):
            op2.Dat(set, 1, name=2)

    def test_dat_illegal_data_access(self, set, backend):
        """Dat initialised without data should raise an exception when
        accessing the data."""
        d = op2.Dat(set, 1)
        with pytest.raises(RuntimeError):
            d.data

    def test_dat_dim(self, set, backend):
        "Dat constructor should create a dim tuple."
        d = op2.Dat(set, 1)
        assert d.dim == (1,)

    def test_dat_dim_list(self, set, backend):
        "Dat constructor should create a dim tuple from a list."
        d = op2.Dat(set, [2,3])
        assert d.dim == (2,3)

    def test_dat_dtype(self, set, backend):
        "Default data type should be numpy.float64."
        d = op2.Dat(set, 1)
        assert d.dtype == np.double

    def test_dat_float(self, set, backend):
        "Data type for float data should be numpy.float64."
        d = op2.Dat(set, 1, [1.0]*set.size)
        assert d.dtype == np.double

    def test_dat_int(self, set, backend):
        "Data type for int data should be numpy.int64."
        d = op2.Dat(set, 1, [1]*set.size)
        assert d.dtype == np.int64

    def test_dat_convert_int_float(self, set, backend):
        "Explicit float type should override NumPy's default choice of int."
        d = op2.Dat(set, 1, [1]*set.size, np.double)
        assert d.dtype == np.float64

    def test_dat_convert_float_int(self, set, backend):
        "Explicit int type should override NumPy's default choice of float."
        d = op2.Dat(set, 1, [1.5]*set.size, np.int32)
        assert d.dtype == np.int32

    def test_dat_illegal_dtype(self, set, backend):
        "Illegal data type should raise DataTypeError."
        with pytest.raises(sequential.DataTypeError):
            op2.Dat(set, 1, dtype='illegal_type')

    @pytest.mark.parametrize("dim", [1, (2,2)])
    def test_dat_illegal_length(self, set, dim, backend):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(sequential.DataValueError):
            op2.Dat(set, dim, [1]*(set.size*np.prod(dim)+1))

    def test_dat_reshape(self, set, backend):
        "Data should be reshaped according to dim."
        d = op2.Dat(set, (2,2), [1.0]*set.size*4)
        assert d.dim == (2,2) and d.data.shape == (set.size,2,2)

    def test_dat_properties(self, set, backend):
        "Dat constructor should correctly set attributes."
        d = op2.Dat(set, (2,2), [1]*set.size*4, 'double', 'bar')
        assert d.dataset == set and d.dim == (2,2) and \
                d.dtype == np.float64 and d.name == 'bar' and \
                d.data.sum() == set.size*4

    def test_dat_soa(self, set, backend):
        "SoA flag should transpose data view"
        d = op2.Dat(set, 2, range(2 * set.size), dtype=np.int32, soa=True)
        expect = np.arange(2 * set.size, dtype=np.int32).reshape(2, 5)
        assert (d.data.shape == expect.shape)

    def test_dat_hdf5(self, h5file, set, backend):
        "Creating a dat from h5file should work"
        d = op2.Dat.fromhdf5(set, h5file, 'dat')
        assert d.dtype == np.float64
        assert d.data.shape == (5,2) and d.data.sum() == 9 * 10 / 2

    def test_data_hdf5_soa(self, h5file, iterset, backend):
        "Creating an SoA dat from h5file should work"
        d = op2.Dat.fromhdf5(iterset, h5file, 'soadat')
        assert d.soa
        assert d.data.shape == (2,5) and d.data.sum() == 9 * 10 / 2

    ## Mat unit tests

    def test_mat_illegal_sets(self, backend):
        "Mat data sets should be a 2-tuple of Sets."
        with pytest.raises(ValueError):
            op2.Mat('illegalset', 1)

    def test_mat_illegal_set_tuple(self, backend):
        "Mat data sets should be a 2-tuple of Sets."
        with pytest.raises(TypeError):
            op2.Mat(('illegalrows', 'illegalcols'), 1)

    def test_mat_illegal_set_triple(self, set, backend):
        "Mat data sets should be a 2-tuple of Sets."
        with pytest.raises(ValueError):
            op2.Mat((set,set,set), 1)

    def test_mat_illegal_dim(self, set, backend):
        "Mat dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Mat((set,set), 'illegaldim')

    def test_mat_illegal_dim_tuple(self, set, backend):
        "Mat dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Mat((set,set), (1,'illegaldim'))

    def test_mat_illegal_name(self, set, backend):
        "Mat name should be string."
        with pytest.raises(sequential.NameTypeError):
            op2.Mat((set,set), 1, name=2)

    def test_mat_sets(self, iterset, dataset, backend):
        "Mat constructor should preserve order of row and column sets."
        m = op2.Mat((iterset, dataset), 1)
        assert m.datasets == (iterset, dataset)

    def test_mat_dim(self, set, backend):
        "Mat constructor should create a dim tuple."
        m = op2.Mat((set,set), 1)
        assert m.dim == (1,)

    def test_mat_dim_list(self, set, backend):
        "Mat constructor should create a dim tuple from a list."
        m = op2.Mat((set,set), [2,3])
        assert m.dim == (2,3)

    def test_mat_dtype(self, set, backend):
        "Default data type should be numpy.float64."
        m = op2.Mat((set,set), 1)
        assert m.dtype == np.double

    def test_dat_properties(self, set, backend):
        "Mat constructor should correctly set attributes."
        m = op2.Mat((set,set), (2,2), 'double', 'bar')
        assert m.datasets == (set,set) and m.dim == (2,2) and \
                m.dtype == np.float64 and m.name == 'bar'

    ## Const unit tests

    def test_const_illegal_dim(self, backend):
        "Const dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Const('illegaldim', 1, 'test_const_illegal_dim')

    def test_const_illegal_dim_tuple(self, backend):
        "Const dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Const((1,'illegaldim'), 1, 'test_const_illegal_dim_tuple')

    def test_const_illegal_data(self, backend):
        "Passing None for Const data should not be allowed."
        with pytest.raises(sequential.DataValueError):
            op2.Const(1, None, 'test_const_illegal_data')

    def test_const_nonunique_name(self, const, backend):
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
        with pytest.raises(sequential.NameTypeError):
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
        "Data type for int data should be numpy.int64."
        c = op2.Const(1, 1, 'test_const_int')
        c.remove_from_namespace()
        assert c.dtype == np.int64

    def test_const_convert_int_float(self, backend):
        "Explicit float type should override NumPy's default choice of int."
        c = op2.Const(1, 1, 'test_const_convert_int_float', 'double')
        c.remove_from_namespace()
        assert c.dtype == np.float64

    def test_const_convert_float_int(self, backend):
        "Explicit int type should override NumPy's default choice of float."
        c = op2.Const(1, 1.5, 'test_const_convert_float_int', 'int')
        c.remove_from_namespace()
        assert c.dtype == np.int64

    def test_const_illegal_dtype(self, backend):
        "Illegal data type should raise DataValueError."
        with pytest.raises(sequential.DataValueError):
            op2.Const(1, 'illegal_type', 'test_const_illegal_dtype', 'double')

    @pytest.mark.parametrize("dim", [1, (2,2)])
    def test_const_illegal_length(self, dim, backend):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(sequential.DataValueError):
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

    def test_const_hdf5(self, h5file, backend):
        "Constant should be correctly populated from hdf5 file."
        c = op2.Const.fromhdf5(h5file, 'constant')
        c.remove_from_namespace()
        assert c.data.sum() == 3
        assert c.dim == (3,)

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
        with pytest.raises(sequential.DataValueError):
            c.data = [1, 2]

    ## Global unit tests

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
        with pytest.raises(sequential.NameTypeError):
            op2.Global(1, 1, name=2)

    def test_global_illegal_data(self, backend):
        "Passing None for Global data should not be allowed."
        with pytest.raises(sequential.DataValueError):
            op2.Global(1, None)

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
        "Data type for int data should be numpy.int64."
        g = op2.Global(1, 1)
        assert g.dtype == np.int64

    def test_global_convert_int_float(self, backend):
        "Explicit float type should override NumPy's default choice of int."
        g = op2.Global(1, 1, 'double')
        assert g.dtype == np.float64

    def test_global_convert_float_int(self, backend):
        "Explicit int type should override NumPy's default choice of float."
        g = op2.Global(1, 1.5, 'int')
        assert g.dtype == np.int64

    def test_global_illegal_dtype(self, backend):
        "Illegal data type should raise DataValueError."
        with pytest.raises(sequential.DataValueError):
            op2.Global(1, 'illegal_type', 'double')

    @pytest.mark.parametrize("dim", [1, (2,2)])
    def test_global_illegal_length(self, dim, backend):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(sequential.DataValueError):
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
        with pytest.raises(sequential.DataValueError):
            c.data = [1, 2]

    ## Map unit tests

    def test_map_illegal_iterset(self, set, backend):
        "Map iterset should be Set."
        with pytest.raises(sequential.SetTypeError):
            op2.Map('illegalset', set, 1, [])

    def test_map_illegal_dataset(self, set, backend):
        "Map dataset should be Set."
        with pytest.raises(sequential.SetTypeError):
            op2.Map(set, 'illegalset', 1, [])

    def test_map_illegal_dim(self, set, backend):
        "Map dim should be int."
        with pytest.raises(sequential.DimTypeError):
            op2.Map(set, set, 'illegaldim', [])

    def test_map_illegal_dim_tuple(self, set, backend):
        "Map dim should not be a tuple."
        with pytest.raises(sequential.DimTypeError):
            op2.Map(set, set, (2,2), [])

    def test_map_illegal_name(self, set, backend):
        "Map name should be string."
        with pytest.raises(sequential.NameTypeError):
            op2.Map(set, set, 1, [], name=2)

    def test_map_illegal_dtype(self, set, backend):
        "Illegal data type should raise DataValueError."
        with pytest.raises(sequential.DataValueError):
            op2.Map(set, set, 1, 'abcdefg')

    def test_map_illegal_length(self, iterset, dataset, backend):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(sequential.DataValueError):
            op2.Map(iterset, dataset, 1, [1]*(iterset.size+1))

    def test_map_convert_float_int(self, iterset, dataset, backend):
        "Float data should be implicitely converted to int."
        m = op2.Map(iterset, dataset, 1, [1.5]*iterset.size)
        assert m.dtype == np.int32 and m.values.sum() == iterset.size

    def test_map_reshape(self, iterset, dataset, backend):
        "Data should be reshaped according to dim."
        m = op2.Map(iterset, dataset, 2, [1]*2*iterset.size)
        assert m.dim == 2 and m.values.shape == (iterset.size,2)

    def test_map_properties(self, iterset, dataset, backend):
        "Data constructor should correctly set attributes."
        m = op2.Map(iterset, dataset, 2, [1]*2*iterset.size, 'bar')
        assert m.iterset == iterset and m.dataset == dataset and m.dim == 2 \
                and m.values.sum() == 2*iterset.size and m.name == 'bar'

    def test_map_hdf5(self, iterset, dataset, h5file, backend):
        "Should be able to create Map from hdf5 file."
        m = op2.Map.fromhdf5(iterset, dataset, h5file, name="map")
        assert m.iterset == iterset
        assert m.dataset == dataset
        assert m.dim == 2
        assert m.values.sum() == sum((1, 2, 2, 3))
        assert m.name == 'map'

    ## IterationSpace unit tests

    def test_iteration_space_illegal_iterset(self, set, backend):
        "IterationSpace iterset should be Set."
        with pytest.raises(sequential.SetTypeError):
            op2.IterationSpace('illegalset', 1)

    def test_iteration_space_illegal_extents(self, set, backend):
        "IterationSpace extents should be int or int tuple."
        with pytest.raises(TypeError):
            op2.IterationSpace(set, 'illegalextents')

    def test_iteration_space_illegal_extents_tuple(self, set, backend):
        "IterationSpace extents should be int or int tuple."
        with pytest.raises(TypeError):
            op2.IterationSpace(set, (1,'illegalextents'))

    def test_iteration_space_extents(self, set, backend):
        "IterationSpace constructor should create a extents tuple."
        m = op2.IterationSpace(set, 1)
        assert m.extents == (1,)

    def test_iteration_space_extents_list(self, set, backend):
        "IterationSpace constructor should create a extents tuple from a list."
        m = op2.IterationSpace(set, [2,3])
        assert m.extents == (2,3)

    def test_iteration_space_properties(self, set, backend):
        "IterationSpace constructor should correctly set attributes."
        i = op2.IterationSpace(set, (2,3))
        assert i.iterset == set and i.extents == (2,3)

    ## Kernel unit tests

    def test_kernel_illegal_name(self, backend):
        "Kernel name should be string."
        with pytest.raises(sequential.NameTypeError):
            op2.Kernel("", name=2)

    def test_kernel_properties(self, backend):
        "Kernel constructor should correctly set attributes."
        k = op2.Kernel("", 'foo')
        assert k.name == 'foo'

class TestBackendAPI:
    """
    Backend API Unit Tests
    """

    @pytest.mark.parametrize("mode", sequential.Access._modes)
    def test_access(self, mode):
        "Access repr should have the expected format."
        a = sequential.Access(mode)
        assert repr(a) == "Access('%s')" % mode

    def test_illegal_access(self):
        "Illegal access modes should raise an exception."
        with pytest.raises(sequential.ModeValueError):
            sequential.Access('ILLEGAL_ACCESS')

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
