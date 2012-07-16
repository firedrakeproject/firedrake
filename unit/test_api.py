import pytest

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
        "Set size should be int"
        with pytest.raises(sequential.SizeTypeError):
            op2.Set('foo')

    def test_set_illegal_name(self):
        "Set name should be string"
        with pytest.raises(sequential.NameTypeError):
            op2.Set(1,2)

    def test_set_size(self, set):
        "Set constructor should correctly set the size"
        assert set.size == 5

    def test_set_repr(self, set):
        "Set repr should have the expected format"
        assert repr(set) == "Set(5, 'foo')"

    def test_set_str(self, set):
        "Set string representation should have the expected format"
        assert str(set) == "OP2 Set: foo with size 5"

    # FIXME: test Set._lib_handle

class TestBackendAPI:
    """
    Backend API Unit Tests
    """

    @pytest.mark.parametrize("mode", sequential.Access._modes)
    def test_access(self, mode):
        a = sequential.Access(mode)
        assert repr(a) == "Access('%s')" % mode

    def test_illegal_access(self):
        with pytest.raises(sequential.ModeValueError):
            sequential.Access('ILLEGAL_ACCESS')
