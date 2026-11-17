import pytest
import numpy as np
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER_PARAMETERS
from itertools import product


@pytest.mark.parametrize(('degree', 'nested', 'elimination'), list(product([1, 2], [True, False], ['0,1', '1,0'])))
def test_lvp_equiv_hdg(degree, nested, elimination):
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

    a = ((inner(q, v) - inner(u, div(v)))*dx
         + inner(uhat('+'), jump(v, n=n))*dS
         + inner(uhat, dot(v, n))*ds
         - inner(q, grad(w))*dx
         + inner(jump(qhat, n=n), w('+'))*dS
         + inner(dot(qhat, n), w)*ds
         + inner(jump(qhat, n=n), mu('+'))*dS
         + inner(uhat, mu)*ds)

    L = inner(f, w)*dx

    params = {
        'mat_type': 'matfree',
        'pmat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.SCPC',
        'pc_sc_eliminate_fields': elimination,
        'condensed_field': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor': DEFAULT_DIRECT_SOLVER_PARAMETERS,
        }
    }

    if nested:
        params['condensed_field']['localsolve'] = {'ksp_type': 'preonly',
                                                   'pc_type': 'fieldsplit',
                                                   'pc_fieldsplit_type': 'schur'}
    ref_problem = LinearVariationalProblem(a, L, s)
    ref_solver = LinearVariationalSolver(ref_problem, solver_parameters=params)
    ref_solver.solve()

    _, __, uhat_ref = s.subfunctions

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
    solver = LinearVariationalSolver(lvp)
    solver.solve()

    assert np.allclose(uhat_ref.dat.data, t.dat.data, rtol=1.E-12)


@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.parametrize(("elimination", "localsolve"),
                         [(("sigma",), {}),
                          (("sigma", "u"), {}),
                          (("u", "sigma"), {}),
                          (("sigma", "u"), {'pc_fieldsplit_type': 'schur'}),
                          (("u", "sigma"), {'pc_fieldsplit_type': 'schur',
                                            'pc_fieldsplit_0_fields': '1',
                                            'pc_fieldsplit_1_fields': '0'})],
                         ids=["sigma", "sigma-u", "u-sigma", "sigma-u-schur", "u-sigma-schur-swapped"])
@pytest.mark.parametrize("order", [("sigma", "u", "lambdar"), ("lambdar", "sigma", "u"), ("sigma", "lambdar", "u")])
def test_scpc_subspace_order(order, elimination, localsolve):
    """Test that SCPC eliminates any group of subspaces, in any order, from a hybridized mixed Poisson problem."""
    mesh = UnitSquareMesh(4, 4)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    U = FunctionSpace(mesh, BrokenElement(FiniteElement("RT", mesh.ufl_cell(), 1)))
    V = FunctionSpace(mesh, "DG", 0)
    T = FunctionSpace(mesh, "HDiv Trace", 0)
    spaces = {"sigma": U, "u": V, "lambdar": T}

    W = MixedFunctionSpace([spaces[name] for name in order])
    trials = dict(zip(order, TrialFunctions(W)))
    tests = dict(zip(order, TestFunctions(W)))
    sigma, u, lambdar = (trials[name] for name in ("sigma", "u", "lambdar"))
    tau, v, gammar = (tests[name] for name in ("sigma", "u", "lambdar"))

    f = sin(pi*x[0])*sin(pi*x[1])
    a = ((inner(sigma, tau) - inner(u, div(tau)) + inner(div(sigma), v))*dx
         + inner(lambdar('+'), jump(tau, n=n))*dS
         + inner(jump(sigma, n=n), gammar('+'))*dS)
    L = inner(f, v)*dx
    bcs = DirichletBC(W.sub(order.index("lambdar")), 0, "on_boundary")

    params = {
        'mat_type': 'matfree',
        'pmat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.SCPC',
        'pc_sc_eliminate_fields': ",".join(str(order.index(name)) for name in elimination),
        'condensed_field': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor': DEFAULT_DIRECT_SOLVER_PARAMETERS,
        }
    }
    if localsolve:
        params['condensed_field']['localsolve'] = {'ksp_type': 'preonly',
                                                   'pc_type': 'fieldsplit',
                                                   **localsolve}
    w = Function(W)
    solve(a == L, w, bcs=bcs, solver_parameters=params)

    w_ref = Function(W)
    solve(a == L, w_ref, bcs=bcs, solver_parameters={'ksp_type': 'preonly',
                                                     'pc_type': 'lu',
                                                     'pc_factor': DEFAULT_DIRECT_SOLVER_PARAMETERS})

    assert np.allclose(w.dat.data_ro, w_ref.dat.data_ro, rtol=1.E-10)
