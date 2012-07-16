from pyop2 import op2

class TestAPI:
    """
    API Unit Tests
    """

    _backend = 'sequential'

    def test_init(self):
        op2.init(self._backend)
        assert op2.backends.get_backend() == 'pyop2.'+self._backend
