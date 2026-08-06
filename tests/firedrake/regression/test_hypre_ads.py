import pytest
from firedrake import *


@pytest.fixture(params=["simplex", "hexahedron"])
def V(request):
    cell = request.param
    if cell == "simplex":
        mesh = UnitCubeMesh(10, 10, 10)
        V = FunctionSpace(mesh, "RT", 1)
    elif cell == "hexahedron":
        mesh = ExtrudedMesh(UnitSquareMesh(10, 10, quadrilateral=True), 10)
        V = FunctionSpace(mesh, "NCF", 1)
    else:
        raise ValueError(f"Unrecognized cell {cell}.")
    return V


@pytest.mark.skiphypre
@pytest.mark.skipcomplex
@pytest.mark.parametrize("mat_type,interface", [("aij", "linear"), ("matfree", "linear"), ("aij", "nonlinear")])
def test_homogeneous_field(V, mat_type, interface):
    u = TrialFunction(V)
    v = TestFunction(V)

    u_exact = Constant((1, 0.5, 4))
    a = inner(div(u), div(v))*dx + inner(u, v)*dx
    L = inner(u_exact, v)*dx

    bc = DirichletBC(V, u_exact, (1, 2, 3, 4))

    params = {
        'snes_type': 'ksponly',
        'mat_type': mat_type,
        'pmat_type': 'aij',
        'ksp_type': 'cg',
        'ksp_max_it': '30',
        'ksp_rtol': '2e-15',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HypreADS',
    }

    u = Function(V)
    solve(a == L, u, bc, solver_parameters=params)
    assert (errornorm(u_exact, u, 'L2') < 1e-10)


@pytest.mark.skiphypre
@pytest.mark.skipcomplex
def test_homogeneous_field_linear_convergence():
    mesh = UnitCubeMesh(10, 10, 10)
    V = FunctionSpace(mesh, "RT", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(div(u), div(v))*dx + inner(u, v)*dx
    L = inner(Constant((1, 0.5, 4)), v)*dx

    bc = DirichletBC(V, Constant((1, 0.5, 4)), (1, 2, 3, 4))

    A = Function(V)
    problem = LinearVariationalProblem(a, L, A, bcs=bc)

    # test hypre options
    for cycle_type in (1, 13):
        expected = 7 if cycle_type == 1 else 9
        params = {'snes_type': 'ksponly',
                  'ksp_type': 'cg',
                  'ksp_max_it': '30',
                  'ksp_rtol': '1e-8',
                  'pc_type': 'python',
                  'pc_python_type': 'firedrake.HypreADS',
                  'hypre_ads_pc_hypre_ads_cycle_type': cycle_type,
                  }

        A.assign(0)
        solver = LinearVariationalSolver(problem, solver_parameters=params)
        solver.solve()
        assert solver.snes.ksp.getIterationNumber() == expected


@pytest.mark.skiphypre
@pytest.mark.skipcomplex
def test_hypre_ads_fieldsplit():
    mesh = UnitCubeMesh(6, 6, 6)
    V = FunctionSpace(mesh, "RT", 1)
    W = V * V * V
    sigma = TrialFunctions(W)
    tau = TestFunctions(W)

    a = inner(sigma, tau) * dx + inner(div(sigma), div(tau)) * dx
    rg = RandomGenerator(PCG64(seed=0))
    L = rg.uniform(W.dual())
    sol = Function(W)

    # Configure fieldsplit with Hypre ADS for each block
    params = {
        "mat_type": "nest",
        "ksp_type": "cg",
        "ksp_rtol": 1e-8,
        "ksp_monitor": None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_ksp_type": "preonly",
        "fieldsplit_pc_type": "python",
        "fieldsplit_pc_python_type": "firedrake.HypreADS",
    }

    prob = LinearVariationalProblem(a, L, sol)
    solver = LinearVariationalSolver(prob, solver_parameters=params)
    solver.solve()

    # Check the condition number
    # ew = solver.snes.ksp.computeEigenvalues().real
    # kappa = 1.0
    # if len(ew):
    #     assert np.isclose(min(ew), 1.0, rtol=1.e-2)
    #     kappa = max(abs(ew)) / min(abs(ew))
    # return kappa ** 0.5
