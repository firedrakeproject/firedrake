import ctypes
from ctypes import POINTER, c_int, c_double

__all__ = ['cFunction']


class _CFunction(ctypes.Structure):
    _fields_ = [("n_cells", c_int),
                ("coords", POINTER(c_double)),
                ("coords_map", POINTER(c_int)),
                ("f", POINTER(c_double)),
                ("f_map", POINTER(c_int))]


def cFunction(function):
    # Retrieve data from Python object
    function_space = function.function_space()
    mesh = function_space.mesh()
    coordinates = mesh.coordinates
    coordinates_space = coordinates.function_space()

    # Store data into ``C struct''
    c_function = _CFunction()
    c_function.n_cells = mesh.num_cells()
    c_function.coords = coordinates.dat.data.ctypes.data_as(POINTER(c_double))
    c_function.coords_map = coordinates_space.cell_node_list.ctypes.data_as(POINTER(c_int))
    c_function.f = function.dat.data.ctypes.data_as(POINTER(c_double))
    c_function.f_map = function_space.cell_node_list.ctypes.data_as(POINTER(c_int))

    # Return pointer
    return ctypes.pointer(c_function)
