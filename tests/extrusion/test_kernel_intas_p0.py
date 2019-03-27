import pytest
import numpy as np
from firedrake import *
import pyop2 as op2


def integrate_assemble_p0(family, degree):
    power = 5
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 10

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.
    # TODO: provide a kernel which will describe how coordinates are extruded.

    mesh = ExtrudedMesh(m, layers, layer_height=0.1)

    fs = FunctionSpace(mesh, family, degree, name="fs")

    f = Function(fs)

    fs1 = FunctionSpace(mesh, family, degree, name="fs1")

    f_rhs = Function(fs1)

    populate_p0 = op2.Kernel("""
void populate_tracer(double *x[], double *c[])
{
  x[0][0] = (c[1][2] + c[0][2]) / 2;
}""", "populate_tracer")

    coords = f.function_space().mesh().coordinates

    op2.par_loop(populate_p0, f.cell_set,
                 f.dat(op2.INC, f.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))

    volume = op2.Kernel("""
void comp_vol(double *rhs[], double *x[], double *y[])
{
  double area = x[0][0]*(x[2][1]-x[4][1]) + x[2][0]*(x[4][1]-x[0][1])
               + x[4][0]*(x[0][1]-x[2][1]);
  if (area < 0)
    area = area * (-1.0);
  rhs[0][0] += 0.5 * area * (x[1][2] - x[0][2]) * y[0][0];
}""", "comp_vol")

    op2.par_loop(volume, f.cell_set,
                 f_rhs.dat(op2.WRITE, f_rhs.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()),
                 f.dat(op2.READ, f.cell_node_map())
                 )

    g = op2.Global(1, data=0.0, name='g')

    reduction = op2.Kernel("""
void comp_reduction(double A[1], double *x[])
{
  A[0] += x[0][0];
}""", "comp_reduction")

    op2.par_loop(reduction, f_rhs.cell_set,
                 g(op2.INC),
                 f_rhs.dat(op2.READ, f_rhs.cell_node_map())
                 )

    return np.abs(g.data[0] - 0.5)


@pytest.mark.parametrize(('family', 'degree'), [('DG', 0)])
def test_firedrake_extrusion_assemble(family, degree):
    assert integrate_assemble_p0(family, degree) < 1.0e-14
