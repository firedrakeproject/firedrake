from firedrake import *
import pytest
import numpy as np


def run_stokes_mini(nest, n):
    length = 10
    mesh = RectangleMesh(2**n, 2**n, length, 1)

    P1 = FiniteElement("CG", cell=mesh.ufl_cell(), degree=1)
    B = FiniteElement("B", cell=mesh.ufl_cell(), degree=3)
    mini = P1 + B
    V = VectorFunctionSpace(mesh, mini)
    P = FunctionSpace(mesh, 'CG', 1)

    W = V*P

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    a = inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx

    f = Constant((0, 0))
    L = inner(f, v)*dx

    # No-slip velocity boundary condition on top and bottom,
    # y == 0 and y == 1
    noslip = Constant((0, 0))
    bc0 = [DirichletBC(W[0], noslip, (3, 4))]

    # Parabolic inflow y(1-y) at x = 0 in positive x direction
    inflow = Expression(("x[1]*(1 - x[1])", "0.0"))
    bc1 = DirichletBC(W[0], inflow, 1)

    # Zero pressure at outlow at x = 1
    bc2 = DirichletBC(W[1], 0.0, 2)

    bcs = bc0 + [bc1, bc2]

    w = Function(W)

    u, p = w.split()

    solve(a == L, w, bcs=bcs,
          solver_parameters={'pc_type': 'fieldsplit',
                             'ksp_rtol': 1e-15,
                             'pc_fieldsplit_type': 'schur',
                             'fieldsplit_schur_fact_type': 'diag',
                             'fieldsplit_0_pc_type': 'redundant',
                             'fieldsplit_0_redundant_pc_type': 'lu',
                             'fieldsplit_1_pc_type': 'none',
                             'ksp_monitor_true_residual': True},
          nest=nest)

    # We've set up Poiseuille flow, so we expect a parabolic velocity
    # field and a linearly decreasing pressure.
    # Need to project the velocity expression because the mini element is not nodal!
    uexact = Function(V).project(Expression(("x[1]*(1 - x[1])", "0.0")))
    pexact = Function(P).interpolate(Expression("2*(L - x[0])", L=length))

    return errornorm(u, uexact, degree_rise=0), errornorm(p, pexact, degree_rise=0)


@pytest.mark.parametrize('nest', [True, False])
def test_stokes_mini(nest):
    u_err = []
    p_err = []

    for n in range(3, 6):
        errs = run_stokes_mini(nest, n)
        u_err.append(errs[0])
        p_err.append(errs[1])

    u_err = np.asarray(u_err)
    p_err = np.asarray(p_err)
    assert (np.log2(u_err[:-1] / u_err[1:]) > 2).all()
    assert (np.log2(p_err[:-1] / p_err[1:]) > 1.5).all()


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
