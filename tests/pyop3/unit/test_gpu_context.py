import pytest
import numpy as np

try:
    import cupy as cp
except ImportError as err:
    pytest.exit("CuPy not available, skipping GPU tests...")


import pyop3 as op3
from firedrake import Function, FunctionSpace, UnitSquareMesh


HOST = op3.HOST_DEVICE
CUDAGPU = op3.CUDAGPU()

STATE_NOT_CREATED = -1
STATE_UNTOUCHED = 0
STATE_MODIFIED = 1

@pytest.fixture()
def mesh():
    return UnitSquareMesh(3, 3)

@pytest.fixture()
def V(mesh):
    return FunctionSpace(mesh, "P", 2)

@pytest.fixture()
def f(V):
    return Function(V)

@pytest.fixture()
def g(V):
    return Function(V)

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

# NOTE: `pytest.fixture`s not used for Offloading GPU tests due to segfault
# Unsure what is causing but we are leaving for now.
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

    def test_host_state_untouched_after_gpu_assign(self):
        """g was not modified on host"""
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, "P", 2)

        f = Function(V).assign(10)
        g = Function(V)
        with op3.offloading(CUDAGPU):
            g.dat.assign(2 * f.dat + 3, eager=True, eager_strategy="array")
        assert state(g, HOST) == 0 

    def test_gpu_state_modified_after_assign(self):
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, "P", 2)

        f = Function(V).assign(10)
        g = Function(V)
        with op3.offloading(CUDAGPU):
            g.dat.assign(2 * f.dat + 3, eager=True, eager_strategy="array")
        assert state(g, CUDAGPU) == 1

class TestOffloadingArraysUpdated:

    def test_gpu_array_modified(self):
        '''Data on GPU is updated in GPU context'''
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, "P", 2)

        f = Function(V).assign(10)
        g = Function(V)
        with op3.offloading(CUDAGPU):
            g.dat.assign(2 * f.dat + 3, eager=True, eager_strategy="array")
            assert (g.dat.data_ro == 23).all()

    def test_gpu_array_modified(self):
        ''' Data on CPU is updated when in CPU context'''
        mesh = UnitSquareMesh(3, 3)
        V = FunctionSpace(mesh, "P", 2)

        f = Function(V).assign(10)
        g = Function(V)
        with op3.offloading(CUDAGPU):
            g.dat.assign(2 * f.dat + 3, eager=True, eager_strategy="array")
        assert (g.dat.data_ro == 23).all()
