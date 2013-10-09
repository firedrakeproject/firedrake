from firedrake import *
import numpy as np

power = 1
m = UnitSquareMesh(2 ** power, 2 ** power)
layers = 11

# Populate the coordinates of the extruded mesh by providing the
# coordinates as a field.

mesh = firedrake.ExtrudedMesh(m, layers, layer_height=0.1)


def identity_xtr(family, degree):
    fs = firedrake.FunctionSpace(mesh, family, degree, name="fs")

    f = firedrake.Function(fs)
    out = firedrake.Function(fs)

    u = firedrake.TrialFunction(fs)
    v = firedrake.TestFunction(fs)

    firedrake.assemble(firedrake.dot(firedrake.grad(u), firedrake.grad(v)) *
                       firedrake.dx)

    f.interpolate(firedrake.Expression("6.0"))

    firedrake.assemble(f * v * firedrake.dx)

    firedrake.solve(u * v * firedrake.dx == f * v * firedrake.dx, out)

    return np.max(np.abs(out.dat.data - f.dat.data))


def run_test():
    family = "Lagrange"
    degree = range(1, 5)

    error = []
    for d in degree:
        error.append(identity_xtr(family, d))
    return error


if __name__ == "__main__":

    error = run_test()
    for i, err in enumerate(error):
        print "Inf norm error in identity for Lagrange %r: %r" % (i + 1, err)
