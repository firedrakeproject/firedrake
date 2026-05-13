"""
Useful links:

  * https://github.com/firedrakeproject/firedrake/blob/main/.github/workflows/core.yml#L476

    How to build a GPU-enabled Firedrake.

  * https://github.com/firedrakeproject/firedrake/blob/connorjward/pyop3-gpu/pyop3/device.py

    An implementation of the 'device' context manager. It needs a big refactor.

  * https://github.com/OP2/PyOP2/pull/691/changes#diff-f8765d963b5adb1788f453e259d8cd45f29cee9670563ddb99b9fe2bba115a12

    Using a wrapper type to track changes between host and device. In pyop3
    this would be the 'ArrayBuffer' object and link into existing
    state tracking.
"""

import numpy as np

from firedrake import *
import pyop3 as op3

from pyop3.device import on_host


# made up API, we need some way to identify the device
host = op3.HOST_DEVICE  # or similar
gpu = op3.CUDAGPU()

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "P", 2)

f = Function(V).assign(10)
g = Function(V)

assert isinstance(f.dat.data_ro, np.ndarray)
assert isinstance(g.dat.data_ro, np.ndarray)

# state tracking checks, .buffer.state is now device-specific
assert f.dat.buffer.state[host] == 1  # modified once
assert f.dat.buffer.state[gpu] == -1  # not created
assert g.dat.buffer.state[host] == 0  # untouched
assert g.dat.buffer.state[gpu] == -1  # not created

with op3.offloading(gpu):
    # Getting the .data attribute on the GPU should give us back a GPU array type
    assert not isinstance(f.dat.data_ro, np.ndarray)
    assert not isinstance(g.dat.data_ro, np.ndarray)

    # # Do the assignment using array operations
    g.dat.assign(2*f.dat + 3, eager=True, eager_strategy="array")

    # # Do the assignment using MLIR (this is a later step)
    # # g.dat.assign(2*f.dat + 3, eager=True, eager_strategy="compile")
    k = Function(V)
    k.dat.buffer.duplicate()
    k.dat.buffer.duplicate(copy=True)

    k.dat.data_rw[...] = 3 

    # state tracking checks
    assert f.dat.buffer.state[host] == 1  # modified once
    assert f.dat.buffer.state[gpu] == 1  # matches host
    assert g.dat.buffer.state[host] == 0  # untouched
    assert g.dat.buffer.state[gpu] == 1  # modified once
    assert k.dat.buffer.state[host] == 0 # not modified
    assert k.dat.buffer.state[gpu] == 1  # modified 

assert isinstance(f.dat.data_ro, np.ndarray)
assert isinstance(g.dat.data_ro, np.ndarray)
assert (g.dat.data_ro == 23).all()
assert (k.dat.data_ro == 3).all()

# state tracking checks
assert f.dat.buffer.state[host] == 1  # modified once
assert f.dat.buffer.state[gpu] == 1  # matches host
assert g.dat.buffer.state[host] == 1  # matches device
assert g.dat.buffer.state[gpu] == 1  # modified once
assert k.dat.buffer.state[host] == 1  # matches device 
assert k.dat.buffer.state[gpu] == 1  # modified once 
