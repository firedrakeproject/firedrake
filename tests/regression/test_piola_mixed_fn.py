import numpy as np
import pytest
from firedrake import *
from tests.common import *


def test_project(mesh):
    U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "N1curl", 1)
    W = U*V

    f = Function(W)
    f.assign(1)

    out = Function(W)
    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)
    f1, f2 = split(f)
    a = dot(u1, v1)*dx + dot(u2, v2)*dx
    L = dot(f1, v1)*dx + dot(f2, v2)*dx

    solve(a == L, out)

    assert np.allclose(out.dat.data[0], f.dat.data[0], rtol=1e-5)
    assert np.allclose(out.dat.data[1], f.dat.data[1], rtol=1e-5)


def test_sphere_project():
    mesh = UnitIcosahedralSphereMesh(0)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))
    U1 = FunctionSpace(mesh, "RT", 1)
    U2 = FunctionSpace(mesh, "CG", 2)
    U3 = FunctionSpace(mesh, "N1curl", 1)
    W = U1*U2*U3

    f = Function(W)
    f1, f2, f3 = f.split()
    f1.assign(1)
    f2.assign(2)
    f3.assign(3)

    out = Function(W)
    u1, u2, u3 = TrialFunctions(W)
    v1, v2, v3 = TestFunctions(W)
    f1, f2, f3 = split(f)
    a = dot(u1, v1)*dx + dot(u2, v2)*dx + dot(u3, v3)*dx
    L = dot(f1, v1)*dx + dot(f2, v2)*dx + dot(f3, v3)*dx

    solve(a == L, out)

    assert np.allclose(out.dat.data[0], f.dat.data[0], rtol=1e-5)
    assert np.allclose(out.dat.data[1], f.dat.data[1], rtol=1e-5)
    assert np.allclose(out.dat.data[2], f.dat.data[2], rtol=1e-5)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
