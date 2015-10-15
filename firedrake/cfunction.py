import ctypes
from ctypes import POINTER, c_int, c_double, c_void_p

from os import path

__all__ = ['cFunction', 'make_c_evaluate']


class _CFunction(ctypes.Structure):
    _fields_ = [("n_cells", c_int),
                ("n_layers", c_int),
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
    c_function.n_layers = mesh.layers - 1 if hasattr(mesh, '_layers') else 1
    c_function.coords = coordinates.dat.data.ctypes.data_as(POINTER(c_double))
    c_function.coords_map = coordinates_space.cell_node_list.ctypes.data_as(POINTER(c_int))
    c_function.f = function.dat.data.ctypes.data_as(POINTER(c_double))
    c_function.f_map = function_space.cell_node_list.ctypes.data_as(POINTER(c_int))
    c_function.sidx = mesh.spatial_index and mesh.spatial_index.ctypes

    # Return pointer
    return ctypes.pointer(c_function)


def make_c_evaluate(function, c_name="evaluate", ldargs=None):
    from pyop2 import compilation, op2
    from pyop2.base import build_itspace
    from pyop2.sequential import generate_cell_wrapper
    from ffc import compile_element

    function_space = function.function_space()
    ufl_element = function_space.ufl_element()
    coordinates = function_space.mesh().coordinates
    coordinates_ufl_element = coordinates.function_space().ufl_element()

    src = compile_element(ufl_element, coordinates_ufl_element, function_space.dim)

    mesh = function_space.mesh()
    coordinates = mesh.coordinates
    arg = coordinates.dat(op2.READ, coordinates.cell_node_map())
    arg.position = 0

    args = (arg,)
    src += generate_cell_wrapper(build_itspace(args, coordinates.cell_set), args,
                                 forward_args=["void*", "double*", "int*"],
                                 kernel_name="to_reference_coords_kernel",
                                 wrapper_name="wrap_to_reference_coords")

    arg = function.dat(op2.READ, function.cell_node_map())
    arg.position = 0

    args = (arg,)
    src += generate_cell_wrapper(build_itspace(args, function.cell_set), args,
                                 forward_args=["double*", "double*"],
                                 kernel_name="evaluate_kernel",
                                 wrapper_name="wrap_evaluate")

    with open(path.join(path.dirname(__file__), "locate.cpp")) as f:
        src += f.read()

    if ldargs is None:
        ldargs = []
    ldargs += ["-lspatialindex"]
    return compilation.load(src, "cpp", c_name, cppargs=["-I%s" % path.dirname(__file__)], ldargs=ldargs)
