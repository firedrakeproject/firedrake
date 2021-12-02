import pytest
import numpy as np
from firedrake import *
import ufl


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


@pytest.fixture
def M(fs):
    uhat = TrialFunction(fs)
    v = TestFunction(fs)
    return inner(uhat, v) * dx


def test_assemble_interp(mesh):
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)

    x, y = SpatialCoordinate(mesh)
    expr = cos(2*pi*x)*sin(2*pi*y)
    f1 = Function(V1).interpolate(expr)

    # Check type
    If1 = Interpolator(f1, V2)
    assert isinstance(If1, ufl.Interp)

    # I(f1, V2)
    a = If1.interpolate()
    b = interpolate(f1, V2)
    assert np.allclose(a.dat.data, b.dat.data)

    assembled_If1 = assemble(If1)
    assert np.allclose(assembled_If1.dat.data, b.dat.data)

    # I(v1, V2)
    v1 = TestFunction(V1)
    Iv1 = Interpolator(v1, V2)

    a = Iv1.interpolate(f1)
    np.allclose(a.dat.data, b.dat.data)

    # Action(I(v1, V2), f1)
    assembled_action_Iv1 = assemble(action(Iv1, f1))
    assert np.allclose(assembled_action_Iv1.dat.data, b.dat.data)

    # fstar = Cofunction(V2)
    # fstar.dat.data[:] = a.dat.data[:]
    # assembled_action_adjoint_Iv1 = assemble(action(adjoint(Iv1), fstar))


def test_solve_interp_f(mesh):
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "DG", 0)
    x, y = SpatialCoordinate(mesh)

    # The space of interpolation (V2) is voluntarily chosen to be of poorer quality than V1.
    # The reasons is that the form is defined on V1 so if we interpolate
    # in a higher-order space we won't see the impact of the interpolation.
    w = TestFunction(V1)
    u = Function(V1)
    f1 = Function(V1).interpolate(cos(x)*sin(y))

    # -- Exact solution with a source term interpolated into DG0
    f2 = interpolate(f1, V2)
    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f2, w)*dx
    solve(F == 0, u)

    # -- Solution where the source term is interpolated via `ufl.Interp`
    u2 = Function(V1)
    If = Interpolator(f1, V2)
    # This requires assembling If
    F2 = inner(grad(w), grad(u2))*dx + inner(u2, w)*dx - inner(If, w)*dx
    solve(F2 == 0, u2)
    assert np.allclose(u.dat.data, u2.dat.data)


def test_solve_interp_u(mesh):
    V1 = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    f = Function(V1).interpolate(cos(x)*sin(y))

    # -- Exact solution
    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    # -- Non mat-free case not supported yet => Need to be able to get the Interpolation matrix -- #
    """
    # -- Solution where the source term is interpolated via `ufl.Interp`
    u2 = Function(V1)
    # Iu is the identity
    Iu = Interpolator(u2, V1)
    # This requires assembling the Jacobian of Iu
    F2 = inner(grad(w), grad(u))*dx + inner(Iu, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)
    """

    # -- Solution where u2 is interpolated via `ufl.Interp` (mat-free)
    u2 = Function(V1)
    # Iu is the identity
    Iu = Interpolator(u2, V1)
    # This requires assembling the action the Jacobian of Iu
    F2 = inner(grad(w), grad(u))*dx + inner(Iu, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2, solver_parameters={"mat_type": "matfree",
                                          "ksp_type": "cg",
                                          "pc_type": "none"})
    assert np.allclose(u.dat.data, u2.dat.data)

    # Same problem with grad(Iu) instead of grad(Iu)
    u2 = Function(V1)
    # Iu is the identity
    Iu = Interpolator(u2, V1)
    # This requires assembling the action the Jacobian of Iu
    F2 = inner(grad(w), grad(Iu))*dx + inner(Iu, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2, solver_parameters={"mat_type": "matfree",
                                          "ksp_type": "cg",
                                          "pc_type": "none"})
    assert np.allclose(u.dat.data, u2.dat.data)
