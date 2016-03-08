import pytest
import numpy as np
from firedrake import *
import pyop2 as op2
import ufl


def integrate_rhs(family, degree):
    power = 5
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 10

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.
    # TODO: provide a kernel which will describe how coordinates are extruded.

    mesh = ExtrudedMesh(m, layers, layer_height=0.1)

    horiz = ufl.FiniteElement(family, "triangle", degree)
    vert = ufl.FiniteElement(family, "interval", degree)
    prod = ufl.TensorProductElement(horiz, vert)

    fs = FunctionSpace(mesh, prod, name="fs")

    f = Function(fs)

    populate_p0 = op2.Kernel("""
void populate_tracer(double *x[], double *c[])
{
  x[0][0] = ((c[1][2] + c[0][2]) / 2);
}""", "populate_tracer")

    coords = f.function_space().mesh().coordinates

    op2.par_loop(populate_p0, f.cell_set,
                 f.dat(op2.INC, f.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))

    g = assemble(f * dx)

    return np.abs(g - 0.5)


def test_firedrake_extrusion_rhs():
    family = "DG"
    degree = 0
    assert integrate_rhs(family, degree) < 1.0e-14

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
