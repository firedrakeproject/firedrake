import pytest
import numpy as np

from pyop2 import op2
from pyop2 import sequential

def pytest_funcarg__set(request):
    return op2.Set(5, 'foo')

class TestUserAPI:
    """
    User API Unit Tests
    """

    _backend = 'sequential'

    ## Init unit tests

    def test_noninit(self):
        "RuntimeError should be raised when using op2 before calling init."
        with pytest.raises(RuntimeError):
            op2.Set(1)

    def test_init(self):
        "init should correctly set the backend."
        op2.init(self._backend)
        assert op2.backends.get_backend() == 'pyop2.'+self._backend

    def test_double_init(self):
        "init should only be callable once."
        with pytest.raises(RuntimeError):
            op2.init(self._backend)

    ## Set unit tests

    def test_set_illegal_size(self):
        "Set size should be int."
        with pytest.raises(sequential.SizeTypeError):
            op2.Set('illegalsize')

    def test_set_illegal_name(self):
        "Set name should be string."
        with pytest.raises(sequential.NameTypeError):
            op2.Set(1,2)

    def test_set_properties(self, set):
        "Set constructor should correctly initialise attributes."
        assert set.size == 5 and set.name == 'foo'

    def test_set_repr(self, set):
        "Set repr should have the expected format."
        assert repr(set) == "Set(5, 'foo')"

    def test_set_str(self, set):
        "Set string representation should have the expected format."
        assert str(set) == "OP2 Set: foo with size 5"

    # FIXME: test Set._lib_handle

    ## Dat unit tests

    def test_dat_illegal_set(self):
        "Dat set should be Set."
        with pytest.raises(sequential.SetTypeError):
            op2.Dat('illegalset', 1)

    def test_dat_illegal_dim(self, set):
        "Dat dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Dat(set, 'illegaldim')

    def test_dat_illegal_dim_tuple(self, set):
        "Dat dim should be int or int tuple."
        with pytest.raises(TypeError):
            op2.Dat(set, (1,'illegaldim'))

    def test_dat_illegal_name(self, set):
        "Dat name should be string."
        with pytest.raises(sequential.NameTypeError):
            op2.Dat(set, 1, name=2)

    def test_dat_illegal_data_access(self, set):
        """Dat initialised without data should raise an exception when
        accessing the data."""
        d = op2.Dat(set, 1)
        with pytest.raises(RuntimeError):
            d.data

    def test_dat_dim(self, set):
        "Dat constructor should create a dim tuple."
        d = op2.Dat(set, 1)
        assert d.dim == (1,)

    def test_dat_dim_list(self, set):
        "Dat constructor should create a dim tuple from a list."
        d = op2.Dat(set, [2,3])
        assert d.dim == (2,3)

    def test_dat_dtype(self, set):
        "Default data type should be numpy.float64."
        d = op2.Dat(set, 1)
        assert d.dtype == np.double

    def test_dat_float(self, set):
        "Data type for float data should be numpy.float64."
        d = op2.Dat(set, 1, [1.0]*set.size)
        assert d.dtype == np.double

    def test_dat_int(self, set):
        "Data type for int data should be numpy.int64."
        d = op2.Dat(set, 1, [1]*set.size)
        assert d.dtype == np.int64

    def test_dat_convert_int_float(self, set):
        "Explicit float type should override NumPy's default choice of int."
        d = op2.Dat(set, 1, [1]*set.size, np.double)
        assert d.dtype == np.float64

    def test_dat_convert_float_int(self, set):
        "Explicit int type should override NumPy's default choice of float."
        d = op2.Dat(set, 1, [1.5]*set.size, np.int32)
        assert d.dtype == np.int32

    def test_dat_illegal_dtype(self, set):
        "Illegal data type should raise DataTypeError."
        with pytest.raises(sequential.DataTypeError):
            op2.Dat(set, 1, dtype='illegal_type')

    @pytest.mark.parametrize("dim", [1, (2,2)])
    def test_dat_illegal_length(self, set, dim):
        "Mismatching data length should raise DataValueError."
        with pytest.raises(sequential.DataValueError):
            op2.Dat(set, dim, [1]*(set.size*np.prod(dim)+1))

    def test_dat_reshape(self, set):
        "Data should be reshaped according to dim."
        d = op2.Dat(set, (2,2), [1.0]*set.size*4)
        assert d.dim == (2,2) and d.data.shape == (set.size,2,2)

    def test_dat_properties(self, set):
        "Dat constructor should correctly set attributes."
        d = op2.Dat(set, (2,2), [1]*set.size*4, 'double', 'bar')
        assert d.dataset == set and d.dim == (2,2) and \
                d.dtype == np.float64 and d.name == 'bar' and \
                d.data.sum() == set.size*4

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
