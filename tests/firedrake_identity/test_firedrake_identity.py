import firedrake
import numpy as np
mesh = firedrake.UnitCubeMesh(1, 1, 1)


def identity(family, degree):
    fs = firedrake.FunctionSpace(mesh, family, degree)

    f = firedrake.Function(fs)
    out = firedrake.Function(fs)

    u = firedrake.TrialFunction(fs)
    v = firedrake.TestFunction(fs)

    firedrake.assemble(u * v * firedrake.dx)

    f.interpolate(firedrake.Expression("x[0]"))

    firedrake.assemble(f * v * firedrake.dx)

    firedrake.solve(u * v * firedrake.dx == f * v * firedrake.dx, out)

    return np.max(np.abs(out.dat.data - f.dat.data))


def run_test():
    family = "Lagrange"
    degree = range(1, 5)

    error = []
    for d in degree:
        error.append(identity(family, d))
    return error


if __name__ == "__main__":

    error = run_test()
    for i, err in enumerate(error):
        print "Inf norm error in identity for Lagrange %r: %r" % (i + 1, err)
