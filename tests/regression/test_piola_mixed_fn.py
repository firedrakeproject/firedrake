import numpy as np
from firedrake import *


def test_project():
    mesh = UnitSquareMesh(5, 5)
    U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "N1curl", 1)
    W = U*V

    f = Function(W)
    f.assign(1)

    out = Function(W)
    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)
    f1, f2 = split(f)
    a = inner(u1, v1)*dx + inner(u2, v2)*dx
    L = inner(f1, v1)*dx + inner(f2, v2)*dx

    solve(a == L, out)

    assert np.allclose(out.dat.data[0], f.dat.data[0], rtol=1e-5)
    assert np.allclose(out.dat.data[1], f.dat.data[1], rtol=1e-5)


def test_sphere_project():
    mesh = UnitIcosahedralSphereMesh(0)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    U1 = FunctionSpace(mesh, "RT", 1)
    U2 = FunctionSpace(mesh, "CG", 2)
    U3 = FunctionSpace(mesh, "N1curl", 1)
    W = U1*U2*U3

    f = Function(W)
    f1, f2, f3 = f.subfunctions
    f1.assign(1)
    f2.assign(2)
    f3.assign(3)

    out = Function(W)
    u1, u2, u3 = TrialFunctions(W)
    v1, v2, v3 = TestFunctions(W)
    f1, f2, f3 = split(f)
    a = inner(u1, v1)*dx + inner(u2, v2)*dx + inner(u3, v3)*dx
    L = inner(f1, v1)*dx + inner(f2, v2)*dx + inner(f3, v3)*dx

    solve(a == L, out)

    assert np.allclose(out.dat.data[0], f.dat.data[0], rtol=1e-5)
    assert np.allclose(out.dat.data[1], f.dat.data[1], rtol=1e-5)
    assert np.allclose(out.dat.data[2], f.dat.data[2], rtol=1e-5)
