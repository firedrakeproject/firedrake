import pytest

from pyop2 import op2

class TestAPI:
    """
    API Unit Tests
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
