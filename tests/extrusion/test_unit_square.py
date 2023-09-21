import numpy as np
from firedrake import *
from firedrake.utils import RealType


def integrate_unit_square(family, degree):
    power = 5
    m = UnitIntervalMesh(2 ** power)
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
    <{0}> area = (real(x[1,1])-real(x[0,1]))*(real(x[2,0])-real(x[0,0]))
    A[0] = A[0] + abs(area)
    """.format(RealType)
    par_loop((domain, instructions), dx, {'A': (g, INC), 'x': (coords, READ)})

    return np.abs(g.dat.data[0] - 1.0)


def test_firedrake_extrusion_unit_square():
    family = "Lagrange"
    degree = 1

    assert integrate_unit_square(family, degree) < 1.0e-12
