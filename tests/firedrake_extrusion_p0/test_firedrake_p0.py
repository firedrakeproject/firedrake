from firedrake import *
import pyop2 as op2

power = 5
m = UnitSquareMesh(2 ** power, 2 ** power)
layers = 11

# Populate the coordinates of the extruded mesh by providing the
# coordinates as a field.
# TODO: provide a kernel which will describe how coordinates are extruded.
extrusion_kernel = """
void extrusion_kernel(double *xtr[], double *x[], int* j[])
{
    //Only the Z-coord is increased, the others stay the same
    xtr[0][0] = x[0][0];
    xtr[0][1] = x[0][1];
    xtr[0][2] = 0.1*j[0][0];
}"""

mesh = firedrake.ExtrudedMesh(m, layers, extrusion_kernel)

#import pyop2.configuration as cfg
# cfg.configure(debug=1)


def integrate_p0(family, degree):
    fs = firedrake.FunctionSpace(mesh, family, degree, name="fs")

    f = firedrake.Function(fs)

    populate_p0 = op2.Kernel("""
void populate_tracer(double *x[])
{
  x[0][0] = 3;
}""", "populate_tracer")

    op2.par_loop(populate_p0, f.cell_set,
                 f.dat(op2.INC, f.cell_node_map))

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

    coords = f.function_space().mesh()._coordinate_field

    op2.par_loop(volume, f.cell_set,
                 g(op2.INC),
                 coords.dat(op2.READ, coords.cell_node_map),
                 f.dat(op2.READ, f.cell_node_map)
                 )

    return np.abs(g.data[0] - 3.0)


def run_test():
    family = "DG"
    degree = 0

    return [integrate_p0(family, degree)]

if __name__ == "__main__":

    result = run_test()
    for i, res in enumerate(result):
        print "Result for extruded unit cube integration %r: %r" % (i + 1, res)
