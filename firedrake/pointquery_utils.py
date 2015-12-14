from __future__ import absolute_import

from os import path

from pyop2 import op2
from pyop2.base import build_itspace
from pyop2.sequential import generate_cell_wrapper

import ffc


def make_args(function):
    arg = function.dat(op2.READ, function.cell_node_map())
    arg.position = 0
    return (arg,)


def make_wrapper(function, **kwargs):
    args = make_args(function)
    return generate_cell_wrapper(build_itspace(args, function.cell_set), args, **kwargs)


def src_locate_cell(mesh):
    src = '#include <evaluate.h>\n'
    src += ffc.compile_coordinate_element(mesh.ufl_coordinate_element())
    src += make_wrapper(mesh.coordinates,
                        forward_args=["void*", "double*", "int*"],
                        kernel_name="to_reference_coords_kernel",
                        wrapper_name="wrap_to_reference_coords")

    with open(path.join(path.dirname(__file__), "locate.cpp")) as f:
        src += f.read()

    return src
