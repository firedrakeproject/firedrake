import numpy as np
from firedrake import *
import pyop2 as op2


def integrate_p0(family, degree):
    power = 5
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 10

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.
    # TODO: provide a kernel which will describe how coordinates are extruded.

    mesh = ExtrudedMesh(m, layers, layer_height=0.1)

    fs = FunctionSpace(mesh, family, degree, name="fs")

    f = Function(fs)

    f.assign(3.0)

    volume = op2.Kernel("""
void comp_vol(double A[1], double *x[], double *y[])
{
  double area = x[0][0]*(x[2][1]-x[4][1]) + x[2][0]*(x[4][1]-x[0][1])
               + x[4][0]*(x[0][1]-x[2][1]);
  if (area < 0)
    area = area * (-1.0);
  A[0] += 0.5 * area * (x[1][2] - x[0][2]) * y[0][0];
}""", "comp_vol")

    g = op2.Global(1, data=0.0, name='g')

    coords = f.function_space().mesh().coordinates

    op2.par_loop(volume, f.cell_set,
                 g(op2.INC),
                 coords.dat(op2.READ, coords.cell_node_map()),
                 f.dat(op2.READ, f.cell_node_map())
                 )

    return np.abs(g.data[0] - 3.0)


def test_firedrake_extrusion_p0():
    family = "DG"
    degree = 0

    assert integrate_p0(family, degree) < 1.0e-11
