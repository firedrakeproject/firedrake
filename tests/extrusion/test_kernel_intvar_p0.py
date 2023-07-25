import numpy as np
from firedrake import *
from firedrake.utils import RealType


def integrate_var_p0(family, degree):
    power = 5
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 10

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.
    # TODO: provide a kernel which will describe how coordinates are extruded.

    mesh = ExtrudedMesh(m, layers, layer_height=0.1)

    fs = FunctionSpace(mesh, family, degree, name="fs")
    f = Function(fs)

    gs = FunctionSpace(mesh, "Real", 0)
    g = Function(gs)

    coords = f.function_space().mesh().coordinates

    domain = ""
    instructions = """
    x[0] = 0.5 * (c[1,2] + c[0,2])
    """
    par_loop((domain, instructions), dx, {'x': (f, INC), 'c': (coords, READ)},
             kernel_kwargs={"requires_zeroed_output_arguments": True})

    instructions = """
    <{0}> area = real(x[0,0])*(real(x[2,1])-real(x[4,1])) + real(x[2,0])*(real(x[4,1])-real(x[0,1])) + real(x[4,0])*(real(x[0,1])-real(x[2,1]))
    A[0] = A[0] + 0.5*abs(area)*(real(x[1,2])-real(x[0,2]))*real(y[0])
    """.format(RealType)
    par_loop((domain, instructions), dx, {'A': (g, INC), 'x': (coords, READ), 'y': (f, READ)},
             kernel_kwargs={"requires_zeroed_output_arguments": True})

    return np.abs(g.dat.data[0] - 0.5)


def test_firedrake_extrusion_var_p0():
    family = "DG"
    degree = 0

    assert integrate_var_p0(family, degree) < 1.0e-13
