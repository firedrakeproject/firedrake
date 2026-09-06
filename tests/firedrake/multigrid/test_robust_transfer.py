import pytest
from firedrake import *


@pytest.fixture
def hierarchy():
    distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    nx = 4
    refine = 3
    base = UnitSquareMesh(nx, nx, distribution_parameters=distribution_parameters)
    mh = MeshHierarchy(base, refine, coarse_facet_label=1000)
    return mh


@pytest.fixture
def mesh(hierarchy):
    return hierarchy[-1]


@pytest.fixture
def V(mesh):
    degree = mesh.topological_dimension
    V = VectorFunctionSpace(mesh, "CG", degree, variant="alfeld")
    return V


@pytest.fixture
def solver(V):
    uh = Function(V)
    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(V.mesh())
    uexact = x * sum(x)

    mu = Constant(1)
    lam = Constant(1E4)
    eps = lambda u: sym(grad(u))
    a = inner(2*mu*eps(u), eps(v))*dx + inner(lam*div(u), div(v))*dx
    L = a(v, uexact)
    bcs = DirichletBC(V, uexact, "on_boundary")

    solver_parameters = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "ksp_rtol": 1e-8,
        "ksp_monitor": None,
        "pc_type": "mg",
        "mg_levels": {
            "ksp_type": "chebyshev",
            "ksp_max_it": 2,
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star_sub_sub_pc_type": "cholesky",
            "pc_star_sub_sub_pc_factor_mat_solver_type": "petsc",
            "pc_star_mat_ordering_type": "nd",
            "pc_star_use_coloring": True,
        },
        "mg_coarse": {
            "mat_type": "aij",
            "pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps",
        }
    }

    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=solver_parameters)
    return solver


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("create_transfer", [CoarsePatchTransferManager, FinePatchTransferManager])
def test_robust_transfer(solver, create_transfer):
    tm = create_transfer()
    u = solver._problem.u
    u.zero()
    solver.set_transfer_manager(tm)
    solver.solve()
    assert solver.snes.ksp.getIterationNumber() < 15
