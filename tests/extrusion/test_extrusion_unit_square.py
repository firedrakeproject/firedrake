import pytest
import numpy as np
from firedrake import *
import pyop2 as op2
from pyop2.profiling import *


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

    area = op2.Kernel("""
void comp_area(double A[1], double *x[], double *y[])
{
  double area = (x[1][1]-x[0][1])*(x[2][0]-x[0][0]);
  if (area < 0)
    area = area * (-1.0);
  A[0] += area;
}""", "comp_area")

    g = op2.Global(1, data=0.0, name='g')

    coords = f.function_space().mesh().coordinates

    op2.par_loop(area, f.cell_set,
                 g(op2.INC),
                 coords.dat(op2.READ, coords.cell_node_map()),
                 f.dat(op2.READ, f.cell_node_map())
                 )

    return np.abs(g.data[0] - 1.0)


def test_firedrake_extrusion_unit_square():
    family = "Lagrange"
    degree = 1

    assert integrate_unit_square(family, degree) < 1.0e-12

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
