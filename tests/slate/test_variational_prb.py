import pytest
import numpy as np
from firedrake import *


@pytest.mark.parametrize('degree', [1, 2])
def test_lvp_equiv_hdg(degree):
    """Runs an HDG problem and checks that passing
    a Slate-defined problem into a variational problem
    produces the same result for the traces as solving
    the whole system using built-in and tested solvers
    and preconditioners.
    """

    mesh = UnitSquareMesh(5, 5)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    U = VectorFunctionSpace(mesh, "DG", degree)
    V = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace", degree)

    W = U * V * T
    s = Function(W)
    q, u, uhat = TrialFunctions(W)
    v, w, mu = TestFunctions(W)

    f = Function(V).interpolate(-div(grad(sin(pi*x[0])*sin(pi*x[1]))))

    tau = Constant(1)
    qhat = q + tau*(u - uhat)*n

    a = ((dot(v, q) - div(v)*u)*dx
         + uhat('+')*jump(v, n=n)*dS
         + uhat*dot(v, n)*ds
         - dot(grad(w), q)*dx
         + jump(qhat, n=n)*w('+')*dS
         + dot(qhat, n)*w*ds
         + mu('+')*jump(qhat, n=n)*dS
         + mu*uhat*ds)

    L = w*f*dx

    params = {'mat_type': 'matfree',
              'pmat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.SCPC',
              'pc_sc_eliminate_fields': '0, 1',
              'condensed_field': {'ksp_type': 'preonly',
                                  'pc_type': 'lu',
                                  'pc_factor_mat_solver_type': 'mumps',
                                  'mat_mumps_icntl_14': 200}}
    ref_problem = LinearVariationalProblem(a, L, s)
    ref_solver = LinearVariationalSolver(ref_problem, solver_parameters=params)
    ref_solver.solve()

    _, __, uhat_ref = s.split()

    # Now using Slate expressions only
    _O = Tensor(a)
    O = _O.blocks

    M = O[:2, :2]
    K = O[:2, 2]
    Q = O[2, :2]
    J = O[2, 2]

    S = J - Q * M.inv * K
    l = assemble(L)
    _R = AssembledVector(l)
    R = _R.blocks
    v1v2 = R[:2]
    v3 = R[2]
    r_lambda = v3 - Q * M.inv * v1v2

    t = Function(T)
    lvp = LinearVariationalProblem(S, r_lambda, t)
    solver = LinearVariationalSolver(lvp, solver_parameters={'ksp_type': 'preonly',
                                                             'pc_type': 'lu',
                                                             'pc_factor_mat_solver_type': 'mumps',
                                                             'mat_mumps_icntl_14': 200})
    solver.solve()

    assert np.allclose(uhat_ref.dat.data, t.dat.data, rtol=1.E-12)
