import pytest
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER
import numpy as np


def run_stabilized_stokes(r, degree, quads):
    """
    Test that we can solve problems involving local projections in Slate.

    We formulate Stokes with equal-order continuous Lagrange elements
    and stabilize the discretizations by adding a mass matrix on the homogeneous
    pressure subspace.

    Reference: arxiv.org/abs/2407.07498
    """
    msh = UnitSquareMesh(2**r, 2**r, quadrilateral=quads)
    V = VectorFunctionSpace(msh, "CG", degree)
    Q = FunctionSpace(msh, "CG", degree)
    Z = V * Q

    u, p = TrialFunctions(Z)
    v, q = TestFunctions(Z)

    # Stokes PDE
    nu = Constant(1)
    gamma = Constant(1E-1)
    a = Tensor((inner(grad(u)*nu, grad(v)) + inner(div(u)*gamma, div(v))) * dx)
    b = Tensor(-inner(div(u), q) * dx)

    # Stabilization on DG space
    X = FunctionSpace(msh, "DG", degree-1)
    tau = TestFunction(X)
    sig = TrialFunction(X)

    h = 1/(nu + gamma)
    mcc = Tensor(h * inner(p, q) * dx)
    mdd = Tensor(h * inner(sig, tau) * dx)
    mdc = Tensor(h * inner(p, tau) * dx)
    mcd = Tensor(h * inner(sig, q) * dx)

    s = mcc - mcd * Inverse(mdd) * mdc

    # Saddle-point bilinear form
    A = a + b.T + b - s

    # Upper-triangular preconditioner
    aP = a + b.T - (mcc + s)

    x, y = SpatialCoordinate(msh)
    bcs = [DirichletBC(Z.sub(0), as_vector([4*y*(1-y), 0]), (1,)),
           DirichletBC(Z.sub(0), 0, (3, 4))]

    solver_parameters = {
        "mat_type": "nest",
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "ksp_max_it": 40,
        "ksp_rtol": 1E-10,
        "ksp_monitor": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_type": "upper",
        "fieldsplit_ksp_type": "preonly",
        "fieldsplit_pc_type": "lu",
        "fieldsplit_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER,
    }
    z = Function(Z)
    L = 0
    problem = LinearVariationalProblem(A, L, z, aP=aP, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    u = z.subfunctions[0]
    return norm(div(u))


@pytest.mark.parametrize(('degree', 'quads', 'rate'),
                         [(1, True, 1.0),
                          (1, False, 1.0),
                          (2, True, 2.0)])
def test_stabilized_stokes(degree, quads, rate):
    diff = np.array([run_stabilized_stokes(r, degree, quads) for r in range(3, 6)])
    conv = np.log2(diff[:-1] / diff[1:])
    tol = 1E-10
    assert (c > rate or d < tol for d, c in zip(diff[1:], conv))
