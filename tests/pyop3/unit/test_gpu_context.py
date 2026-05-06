import pytest
import numpy as np

try:
    import cupy as cp
except ImportError as err:
    pytest.exit("CuPy not available, skipping GPU tests...")


import pyop3 as op3
from firedrake import Function, FunctionSpace, UnitSquareMesh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HOST = op3.HOST_DEVICE
CUDAGPU = op3.CUDAGPU()

STATE_NOT_CREATED = -1
STATE_UNTOUCHED = 0
STATE_MODIFIED = 1

@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(3, 3)

@pytest.fixture(scope="module")
def function_space(mesh):
    return FunctionSpace(mesh, "P", 2)

@pytest.fixture(scope="module")
def f(function_space):
    return Function(function_space)

@pytest.fixture(scope="module")
def g(function_space):
    return Function(function_space)

def state(func, device):
    """Shorthand for reading buffer state on a given device."""
    return func.dat.buffer.state[device]

class TestInitialState:
    def test_host_data_is_numpy(self, f):
        assert isinstance(f.dat.data_ro, np.ndarray)

    def test_host_state_modified(self, f):
        """Assign affects buffer counter on host"""
        f.dat.assign(10, eager=True, eager_strategy="array")
        assert state(f, HOST) == 1

    def test_gpu_state_not_created(self, f):
        """CUDAGPU buffer should not exist before any offloading."""
        assert state(f, CUDAGPU) == STATE_NOT_CREATED

class TestOffloadingArrayTypes:
    """Inside op3.offloading, data array type should be GPU array types"""

    def test_buffer_evaluates_cupy_on_cudagpu(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, "P", 2)

        f = Function(V).assign(10)
        g = Function(V)
        with op3.offloading(CUDAGPU):
            assert not isinstance(f.dat.data_ro, np.ndarray)

    def test_buffer_creation_on_cudagpu(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, "P", 2)

        f = Function(V).assign(10)
        g = Function(V)
        with op3.offloading(CUDAGPU):
            k = Function(V)
            assert not isinstance(k.dat.data_ro, np.ndarray)

class TestOffloadingAssignmentState:
    
    @pytest.fixture(autouse=True)
    def _run_gpu_assign(self, f, g):
        with op3.offloading(CUDAGPU):
            g.dat.assign(2 * f.dat + 3, eager=True, eager_strategy="array")

    def test_host_state_untouched_after_gpu_assign(self, g):
        """g was not modified on host"""
        assert state(g, HOST) == 0 

    def test_g_gpu_state_modified_after_assign(self, g):
        assert state(g, CUDAGPU) == 1
