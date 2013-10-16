import numpy as np
import pytest

from firedrake import *


def identity_xtr(family, degree):
    power = 1
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 11

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.

    mesh = firedrake.ExtrudedMesh(m, layers, layer_height=0.1)

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


def test_firedrake_extrusion_assembly():
    family = "Lagrange"
    degree = range(1, 5)

    error = np.array([identity_xtr(family, d) for d in degree])
    assert (error < np.array([1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8])).all()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
