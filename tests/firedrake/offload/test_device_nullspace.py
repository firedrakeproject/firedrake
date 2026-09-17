from firedrake import *
import pytest


@pytest.mark.skipnogpu
def test_device_nullspace():
    # Tests that a provided near-nullspace vector basis has been offloaded to the device
    # Adapted from test_near_nullspace in test_nullspace.py
    nested_parameters = {
        "pc_type": "ksp",
        "ksp": {
            "ksp_type": "cg",
            "ksp_max_it": 50,
            "ksp_view": None,
            "ksp_rtol": "1e-10",
            "ksp_monitor": None,
            "pc_type": "gamg",
        },
    }
    parameters = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.OffloadPC",
        "offload": nested_parameters,
    }

    mesh = UnitSquareMesh(100, 100)
    x, y = SpatialCoordinate(mesh)
    dim = 2
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    mu = Constant(0.2)
    lmbda = Constant(0.3)

    def sigma(fn):
        return 2.0 * mu * sym(grad(fn)) + lmbda * tr(sym(grad(fn))) * Identity(dim)

    w_exact = Function(V)
    w_exact.interpolate(as_vector([x * y, x * y]))
    f = Constant((mu + lmbda, mu + lmbda))
    F = inner(sigma(u), grad(v)) * dx + inner(f, v) * dx

    bcs = [DirichletBC(V, w_exact, (1, 2, 3, 4))]

    n0 = Constant((1, 0))
    n1 = Constant((0, 1))
    n2 = as_vector([y - 0.5, -(x - 0.5)])
    ns = [n0, n1, n2]
    n_interp = [assemble(interpolate(n, V)) for n in ns]
    nsp = VectorSpaceBasis(vecs=n_interp)
    nsp.orthonormalize()

    w1 = Function(V)
    problem = LinearVariationalProblem(lhs(F), rhs(F), w1, bcs=bcs)
    solver = LinearVariationalSolver(
        problem, solver_parameters=parameters, near_nullspace=nsp
    )
    solver.solve()

    _, P = solver.snes.ksp.pc.getPythonContext().pc.getOperators()
    nns = P.getNearNullSpace()
    assert nns.handle != 0
    for vec in nns.getVecs():
        assert vec.getType() == "seqcuda"
