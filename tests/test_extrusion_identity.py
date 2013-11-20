import numpy as np
import pytest

from firedrake import *


def identity_xtr(family, degree):
    power = 1
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 11

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.
    # TODO: provide a kernel which will describe how coordinates are extruded.

    mesh = firedrake.ExtrudedMesh(m, layers, layer_height=0.1)

    fs = firedrake.FunctionSpace(mesh, family, degree, name="fs")

    f = firedrake.Function(fs)
    out = firedrake.Function(fs)

    u = firedrake.TrialFunction(fs)
    v = firedrake.TestFunction(fs)

    firedrake.assemble(u * v * firedrake.dx)

    f.interpolate(firedrake.Expression("x[0]"))

    firedrake.assemble(f * v * firedrake.dx)

    firedrake.solve(u * v * firedrake.dx == f * v * firedrake.dx, out)

    return np.max(np.abs(out.dat.data - f.dat.data))


def vector_identity(family, degree):
    m = UnitSquareMesh(2, 2)
    layers = 11
    mesh = ExtrudedMesh(m, layers, layer_height=0.1)
    fs = VectorFunctionSpace(mesh, family, degree, name="fs")
    f = Function(fs)
    out = Function(fs)
    u = TrialFunction(fs)
    v = TestFunction(fs)
    f.interpolate(Expression(("x[0]", "x[1]", "x[2]")))
    solve(inner(u, v)*dx == inner(f, v)*dx, out)
    return np.max(np.abs(out.dat.data_ro - f.dat.data_ro))


def test_firedrake_extrusion_identity():
    family = "Lagrange"
    degree = range(1, 5)

    error = np.array([identity_xtr(family, d) for d in degree])
    assert (error < np.array([1.0e-14, 1.0e-6, 1.0e-6, 1.0e-6])).all()


@pytest.mark.xfail(reason="No support for vector function spaces on extruded meshes")
def test_extrusion_vector_identity():
    family = "DG"
    error = np.array([vector_identity(family, d) for d in range(0, 5)])
    assert (error < 1e-6).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
