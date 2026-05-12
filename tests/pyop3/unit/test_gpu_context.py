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
def FuncSpace(mesh):
    return FunctionSpace(mesh, "P", 2)

@pytest.fixture()
def f(FuncSpace):
    return Function(FuncSpace).assign(10)

@pytest.fixture()
def g(FuncSpace):
    return Function(FuncSpace)

def state(func, device):
    """Shorthand for reading buffer state on a given device."""
    return func.dat.buffer.state[device]


class TestInitialState:
    def test_host_data_is_numpy(self, f):
        assert isinstance(f.dat.data_ro, np.ndarray)

    def test_host_state_modified(self, f):
        """Assign affects buffer counter on host"""
        old_state = state(f, HOST)
        f.dat.assign(10, eager=True, eager_strategy="array")
        assert state(f, HOST) == old_state + 1

    def test_gpu_state_not_created(self, f):
        """CUDAGPU buffer should not exist before any offloading."""
        assert state(f, CUDAGPU) == STATE_NOT_CREATED

class TestOffloadingArrayTypes:
    """Inside op3.offloading, data array type should be GPU array types"""

    def test_buffer_evaluates_cupy_on_cudagpu(self, FuncSpace):
        f = Function(FuncSpace).assign(10)
        with op3.offloading(CUDAGPU):
            assert isinstance(f.dat.data_ro, cp.ndarray)

    def test_buffer_creation_on_cudagpu(self, FuncSpace):
        with op3.offloading(CUDAGPU):
            k = Function(FuncSpace)
            assert isinstance(k.dat.data_ro, cp.ndarray)

class TestOffloadingAssignmentState:

    def test_host_state_untouched_after_gpu_assign(self, f, g):
        """g was not modified on host"""
        with op3.offloading(CUDAGPU):
            g.dat.assign(2 * f.dat + 3, eager=True, eager_strategy="array")
        assert state(g, HOST) == 0 

    def test_gpu_state_modified_after_assign(self, f, g):
        """g was modified on CUDAGPU"""
        with op3.offloading(CUDAGPU):
            g.dat.assign(2 * f.dat + 3, eager=True, eager_strategy="array")
        assert state(g, CUDAGPU) == 1

class TestOffloadingArraysUpdated:

    def test_gpu_array_modified(self, g):
        '''Data on GPU is updated in GPU context'''
        with op3.offloading(CUDAGPU):
            g.dat.assign(23, eager=True, eager_strategy="array")
            assert (g.dat.data_ro == 23).all()

    def test_gpu_array_modified_copied_to_host(self, g):
        ''' Data on CPU is updated when in CPU context'''
        with op3.offloading(CUDAGPU):
            g.dat.assign(23, eager=True, eager_strategy="array")
        assert (g.dat.data_ro == 23).all()

    def test_gpu_data_wo_copied_to_host(self, g):
        ''' Data on CPU is updated when in CPU context'''
        with op3.offloading(CUDAGPU):
            g.dat.data_wo[...] = 23
        assert (g.dat.data_ro == 23).all()

class TestDeviceArrayDuplication:
    
    def test_duplicate_not_same(self, FuncSpace):
        """Duplicate buffer is not same object"""
        with op3.offloading(CUDAGPU):
            k = Function(FuncSpace)
            k_dup_buffer = k.dat.buffer.duplicate()
            assert type(k_dup_buffer) == type(k.dat.buffer)
            assert not k_dup_buffer is k.dat.buffer

    def test_duplicate_to_device(self, FuncSpace):
        """ Buffer maintains device context when copied"""
        with op3.offloading(CUDAGPU):
            k = Function(FuncSpace)
            k_dup_buffer = k.dat.buffer.duplicate()
            assert isinstance(k_dup_buffer.data_ro, cp.ndarray)

    def test_duplicate_copy_to_device(self, FuncSpace):
        """ Buffer maintains device context when exact copy"""
        with op3.offloading(CUDAGPU):
            k = Function(FuncSpace)
            k_dup_buffer = k.dat.buffer.duplicate(copy=True)
            assert isinstance(k_dup_buffer.data_ro, cp.ndarray)
