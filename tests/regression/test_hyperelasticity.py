# coding=utf-8
from firedrake import *
import pytest


@pytest.mark.skipif(utils.complex_mode, reason="Not clear this should work in complex")
def test_hyperelastic_convergence():
    # This is a simple neo-Hookean hyperelastic model of a
    # compressed rubber block.  The block is anchored on its
    # bottom boundary and undergoes both compression due to
    # internal body forces (the block is massive) and an external
    # free-slip compression from above.

    # This test does not check for the correctness of the
    # solution, but rather that different COFFEE optimisation
    # levels do not materially affect the convergence behaviour.

    mesh = UnitSquareMesh(1, 1)

    V = VectorFunctionSpace(mesh, "Lagrange", 1)

    bcs = DirichletBC(V, Constant((0, 0)), 3)

    v = TestFunction(V)
    u = Function(V)

    T = Constant((0, -0.5))
    B = Constant((0, -0.25))

    d = u.geometric_dimension()
    I = Identity(d)
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J = det(F)

    # Lamé parameters, "quite squishy"
    mu = Constant(6.3)
    lmbda = Constant(10.0)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    # Total potential energy
    Pi = psi*dx - dot(T, u)*ds(4) - dot(B, u)*dx

    F = derivative(Pi, u, v)

    problem = NonlinearVariationalProblem(F, u, bcs=bcs)

    solver = NonlinearVariationalSolver(problem, solver_parameters={"pc_type": "lu",
                                                                    "snes_atol": 1e-8})

    solver.solve()
    assert solver.snes.getIterationNumber() == 3
