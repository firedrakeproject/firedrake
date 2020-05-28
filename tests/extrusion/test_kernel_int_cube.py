import numpy as np
from firedrake import *


def integrate_unit_cube(family, degree):
    power = 5
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 10

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.
    # A kernel which describes how coordinates are extruded.

    mesh = ExtrudedMesh(m, layers, layer_height=0.1)

    fs = FunctionSpace(mesh, family, degree, name="fs")
    f = Function(fs)
    gs = FunctionSpace(mesh, "Real", 0)
    g = Function(gs)

    coords = f.function_space().mesh().coordinates

    domain = ""
    instructions = """
    <float64> area = x[0,0]*(x[2,1]-x[4,1]) + x[2,0]*(x[4,1]-x[0,1]) + x[4,0]*(x[0,1]-x[2,1])
    A[0] = A[0] + 0.5*abs(area)*(x[1,2]-x[0,2])
    """

    par_loop((domain, instructions), dx, {'A': (g, INC), 'x': (coords, READ)},
             is_loopy_kernel=True)

    return np.abs(g.dat.data[0] - 1.0)


def test_firedrake_extrusion_unit_cube():
    family = "Lagrange"
    degree = 1

    assert integrate_unit_cube(family, degree) < 1.0e-12
