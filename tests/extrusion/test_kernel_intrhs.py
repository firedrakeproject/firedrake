import numpy as np
from firedrake import *
import finat.ufl


def integrate_rhs(family, degree):
    power = 5
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 10

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.
    # TODO: provide a kernel which will describe how coordinates are extruded.

    mesh = ExtrudedMesh(m, layers, layer_height=0.1)

    horiz = finat.ufl.FiniteElement(family, "triangle", degree)
    vert = finat.ufl.FiniteElement(family, "interval", degree)
    prod = finat.ufl.TensorProductElement(horiz, vert)

    fs = FunctionSpace(mesh, prod, name="fs")
    f = Function(fs)

    coords = f.function_space().mesh().coordinates

    domain = ""
    instructions = """
    x[0,0] = 0.5 * (c[1,2] + c[0,2])
    """

    par_loop((domain, instructions), dx, {'x': (f, INC), 'c': (coords, READ)},
             kernel_kwargs={"requires_zeroed_output_arguments": True})

    g = assemble(f * dx)

    return np.abs(g - 0.5)


def test_firedrake_extrusion_rhs():
    family = "DG"
    degree = 0
    assert integrate_rhs(family, degree) < 2.0e-12
