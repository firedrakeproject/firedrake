import pytest
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


def bddc_params(static_condensation):
    chol = {
        "pc_type": "cholesky",
        "pc_factor_mat_solver_type": "petsc",
        "pc_factor_mat_ordering_type": "natural",
    }
    sp = {
        "pc_type": "python",
        "pc_python_type": "firedrake.BDDCPC",
        "bddc_pc_bddc_neumann": chol,
        "bddc_pc_bddc_dirichlet": chol,
        "bddc_pc_bddc_coarse": DEFAULT_DIRECT_SOLVER,
    }
    return sp


def solver_parameters(static_condensation=True):
    rtol = 1E-8
    atol = 1E-12
    sp_bddc = bddc_params(static_condensation)
    repeated = True
    if static_condensation:
        sp = {
            "pc_type": "python",
            "pc_python_type": "firedrake.FacetSplitPC",
            "facet_pc_type": "python",
            "facet_pc_python_type": "firedrake.FDMPC",
            "facet_fdm_static_condensation": True,
            "facet_fdm_pc_use_amat": False,
            "facet_fdm_mat_type": "is",
            "facet_fdm_mat_is_allow_repeated": repeated,
            "facet_fdm_pc_type": "fieldsplit",
            "facet_fdm_pc_fieldsplit_type": "symmetric_multiplicative",
            "facet_fdm_pc_fieldsplit_diag_use_amat": False,
            "facet_fdm_pc_fieldsplit_off_diag_use_amat": False,
            "facet_fdm_fieldsplit_ksp_type": "preonly",
            "facet_fdm_fieldsplit_0_pc_type": "jacobi",
            "facet_fdm_fieldsplit_1": sp_bddc,
        }
    else:
        sp = {
            "pc_type": "python",
            "pc_python_type": "firedrake.FDMPC",
            "fdm_pc_use_amat": False,
            "fdm_mat_type": "is",
            "fdm_mat_is_allow_repeated": repeated,
            "fdm": sp_bddc,
        }
    sp.update({
        "mat_type": "matfree",
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "ksp_monitor": None,
        "ksp_rtol": rtol,
        "ksp_atol": atol,
    })
    return sp


def solve_riesz_map(mesh, family, degree, bcs, condense):
    dirichlet_ids = []
    if bcs:
        dirichlet_ids = ["on_boundary"]
        if hasattr(mesh, "extruded") and mesh.extruded:
            dirichlet_ids.extend(["bottom", "top"])

    tdim = mesh.topological_dimension()
    if family.endswith("E"):
        family = "RTCE" if tdim == 2 else "NCE"
    if family.endswith("F"):
        family = "RTCF" if tdim == 2 else "NCF"
    V = FunctionSpace(mesh, family, degree, variant="fdm")
    v = TestFunction(V)
    u = TrialFunction(V)
    d = {
        H1: grad,
        HCurl: curl,
        HDiv: div,
    }[V.ufl_element().sobolev_space]

    formdegree = V.finat_element.formdegree
    if formdegree == 0:
        a = inner(d(u), d(v)) * dx(degree=2*degree)
    else:
        a = (inner(u, v) + inner(d(u), d(v))) * dx(degree=2*degree)

    rg = RandomGenerator(PCG64(seed=123456789))
    u_exact = rg.uniform(V, -1, 1)
    L = ufl.replace(a, {u: u_exact})
    bcs = [DirichletBC(V, u_exact, sub) for sub in dirichlet_ids]
    nsp = None
    if formdegree == 0:
        nsp = VectorSpaceBasis([Function(V).interpolate(Constant(1))])
        nsp.orthonormalize()

    uh = Function(V, name="solution")
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    sp = solver_parameters(condense)
    solver = LinearVariationalSolver(problem, near_nullspace=nsp,
                                     solver_parameters=sp,
                                     options_prefix="")
    solver.solve()
    return solver.snes.getLinearSolveIterations()


@pytest.fixture(params=(2, 3), ids=("square", "cube"))
def mesh(request):
    nx = 4
    dim = request.param
    msh = UnitSquareMesh(nx, nx, quadrilateral=True)
    if dim == 3:
        msh = ExtrudedMesh(msh, nx)
    return msh


@pytest.mark.parallel
@pytest.mark.parametrize("family", "Q")
@pytest.mark.parametrize("degree", (4,))
@pytest.mark.parametrize("condense", (False, True))
def test_bddc_fdm(mesh, family, degree, condense):
    bcs = True
    tdim = mesh.topological_dimension()
    expected = 6 if tdim == 2 else 11
    assert solve_riesz_map(mesh, family, degree, bcs, condense) <= expected
