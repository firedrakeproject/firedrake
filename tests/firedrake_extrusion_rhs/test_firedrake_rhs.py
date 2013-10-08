from firedrake import *
import pyop2 as op2
import ufl

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


def integrate_rhs(family, degree):
    horiz = ufl.FiniteElement(family, None, degree)
    vert = ufl.FiniteElement(family, None, degree)
    prod = ufl.OuterProductElement(horiz, vert)

    fs = firedrake.FunctionSpace(mesh, prod, name="fs")

    f = firedrake.Function(fs)

    populate_p0 = op2.Kernel("""
void populate_tracer(double *x[], double *c[])
{
  x[0][0] = ((c[1][2] + c[0][2]) / 2);
}""", "populate_tracer")

    coords = f.function_space().mesh()._coordinate_field

    op2.par_loop(populate_p0, f.cell_set,
                 f.dat(op2.INC, f.cell_node_map),
                 coords.dat(op2.READ, coords.cell_node_map))

    g = firedrake.assemble(f * firedrake.dx)

    return np.abs(g - 0.5)


def run_test():
    family = "DG"
    degree = 0

    return [integrate_rhs(family, degree)]

if __name__ == "__main__":

    result = run_test()
    for i, res in enumerate(result):
        print "Result for extruded unit cube integration %r: %r" % (i + 1, res)
