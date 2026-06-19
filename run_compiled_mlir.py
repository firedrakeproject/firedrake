import ctypes
import numpy as np
import cupy as cp

from ctypes import c_void_p, c_longlong, Structure

class MemRefDescriptor(Structure):
    _fields_ = [
        ("allocated", c_void_p),  
        ("aligned", c_void_p),  
        ("offset", c_longlong),
        ("shape", c_longlong * 1),
        ("stride", c_longlong * 1),
    ]

def numpy_to_memref(arr):
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    desc = MemRefDescriptor()
    desc.allocated = arr.ctypes.data_as(c_void_p)
    desc.aligned = desc.allocated
    desc.offset = 0
    desc.shape[0] = arr.shape[0]
    desc.stride[0] = 1

    return desc


if __name__ == "__main__":
    lib = ctypes.CDLL("./liboutput.dylib")

    array_add = lib._mlir_ciface_add
    array_add.argtypes = [
        ctypes.POINTER(MemRefDescriptor)
    ] * 3  
    
    size = 8
    a = np.ones(size, dtype=np.float64)
    b = np.ones(size, dtype=np.float64) * 2
    c = np.zeros(size, dtype=np.float64)

    a_desc = numpy_to_memref(a)
    b_desc = numpy_to_memref(b)
    c_desc = numpy_to_memref(c)

    array_add(ctypes.byref(a_desc), ctypes.byref(b_desc), ctypes.byref(c_desc))

    expected = a + b
    np.testing.assert_array_almost_equal(c, expected)
    print("Array addition successful!")
    print(f"First few elements: {c[:5]}") 
