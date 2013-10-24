import pytest
from firedrake import *


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


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
