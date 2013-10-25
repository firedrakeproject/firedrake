import pytest
from firedrake import *
from numpy.linalg import norm


@pytest.fixture
def a_L_out():
    mesh = UnitCubeMesh(1, 1, 1)
    fs = FunctionSpace(mesh, 'CG', 1)

    f = Function(fs)
    out = Function(fs)

    u = TrialFunction(fs)
    v = TestFunction(fs)
    return u*v*dx, f*v*dx, out


def test_linear_solver_api(a_L_out):
    a, L, out = a_L_out
    p = LinearVariationalProblem(a, L, out)
    solver = LinearVariationalSolver(p, parameters={'ksp_type': 'cg'})

    assert solver.parameters['snes_type'] == 'ksponly'
    assert solver.parameters['ksp_rtol'] == 1e-7
    assert solver.snes.getType() == solver.snes.Type.KSPONLY
    assert solver.snes.getKSP().getType() == solver.snes.getKSP().Type.CG
    rtol, _, _, _ = solver.snes.getKSP().getTolerances()
    assert rtol == solver.parameters['ksp_rtol']

    solver.parameters['ksp_type'] = 'gmres'
    solver.parameters['ksp_rtol'] = 1e-8
    solver.solve()
    assert solver.snes.getKSP().getType() == solver.snes.getKSP().Type.GMRES
    assert solver.parameters['ksp_rtol'] == 1e-8
    rtol, _, _, _ = solver.snes.getKSP().getTolerances()
    assert rtol == solver.parameters['ksp_rtol']


def test_nonlinear_solver_api(a_L_out):
    a, L, out = a_L_out
    J = a
    F = action(a, out) - L
    p = NonlinearVariationalProblem(F, out, J=J)
    solver = NonlinearVariationalSolver(p, parameters={'snes_type': 'ksponly'})

    assert solver.snes.getType() == solver.snes.Type.KSPONLY
    rtol, _, _, _ = solver.snes.getTolerances()
    assert rtol == 1e-8

    solver.parameters['snes_rtol'] = 1e-3
    solver.parameters['snes_type'] = 'newtonls'
    solver.solve()
    assert solver.parameters['snes_rtol'] == 1e-3
    assert solver.snes.getType() == solver.snes.Type.NEWTONLS
    rtol, _, _, _ = solver.snes.getTolerances()
    assert rtol == solver.parameters['snes_rtol']


def test_linear_solves_equivalent():
    """solve(a == L, out) should return the same as solving with the assembled objects.

    This relies on two different code paths agreeing on the same set of solver parameters."""
    mesh = UnitSquareMesh(50, 50)

    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V)
    f.assign(1)
    f.vector()[:] = 1.
    t = TestFunction(V)
    q = TrialFunction(V)

    a = inner(t, q)*dx
    L = inner(f, t)*dx

    # Solve the system using forms
    sol = Function(V)
    solve(a == L, sol)

    # And again
    sol2 = Function(V)
    solve(a == L, sol2)
    assert norm(sol.vector()[:] - sol2.vector()[:]) == 0

    # Solve the system using preassembled objects
    sol3 = Function(V)
    solve(assemble(a), sol3, assemble(L).dat)
    assert norm(sol.vector()[:] - sol3.vector()[:]) == 0


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
