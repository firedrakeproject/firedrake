import ctypes
from ctypes import POINTER, c_int, c_double, c_void_p

from os import path

__all__ = ['cFunction', 'make_c_evaluate']


class _CFunction(ctypes.Structure):
    _fields_ = [("n_cells", c_int),
                ("coords", POINTER(c_double)),
                ("coords_map", POINTER(c_int)),
                ("f", POINTER(c_double)),
                ("f_map", POINTER(c_int)),
                ("sidx", c_void_p)]


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
    c_function.sidx = mesh.spatial_index.ctypes

    # Return pointer
    return ctypes.pointer(c_function)


def make_c_evaluate(function, c_name="evaluate", ldargs=None):
    from pyop2 import compilation
    from ffc import compile_element

    function_space = function.function_space()
    ufl_element = function_space.ufl_element()
    coordinates = function_space.mesh().coordinates
    coordinates_ufl_element = coordinates.function_space().ufl_element()

    src = compile_element(ufl_element, coordinates_ufl_element)

    if ldargs is None:
        ldargs = []
    ldargs += [path.join(path.dirname(__file__), "locate.o"), "-lspatialindex"]
    return compilation.load(src, "c", c_name, cppargs=["-I%s" % path.dirname(__file__)], ldargs=ldargs)
