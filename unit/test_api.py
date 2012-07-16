import pytest

from pyop2 import op2
from pyop2 import sequential

class TestUserAPI:
    """
    User API Unit Tests
    """

    _backend = 'sequential'

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
