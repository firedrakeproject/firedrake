import pytest
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


@pytest.fixture
def rg():
    return RandomGenerator(PCG64(seed=123456789))


def bddc_params():
    chol = {
        "pc_type": "cholesky",
        "pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER,
    }
    sp = {
        "mat_type": "is",
        "pc_type": "python",
        "pc_python_type": "firedrake.BDDCPC",
        "bddc_pc_bddc_neumann": chol,
        "bddc_pc_bddc_dirichlet": chol,
        "bddc_pc_bddc_coarse": chol,
    }
    return sp


def solver_parameters(static_condensation=False, variant=None):
    rtol = 1E-8
    atol = 0
    sp_bddc = bddc_params()
    if variant != "fdm":
        sp = sp_bddc

    elif static_condensation:
        sp = {
            "pc_type": "python",
            "pc_python_type": "firedrake.FacetSplitPC",
            "facet_pc_type": "python",
            "facet_pc_python_type": "firedrake.FDMPC",
            "facet_fdm_static_condensation": True,
            "facet_fdm_pc_use_amat": False,
            "facet_fdm_mat_type": "is",
            "facet_fdm_mat_is_allow_repeated": True,
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
            "fdm_mat_is_allow_repeated": True,
            "fdm": sp_bddc,
        }

    sp.update({
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "ksp_converged_reason": None,
        "ksp_monitor": None,
        "ksp_rtol": rtol,
        "ksp_atol": atol,
    })
    if variant == "fdm":
        sp["mat_type"] = "matfree"
    return sp


def solve_riesz_map(rg, mesh, family, degree, variant, bcs, condense=False, vector=False):
    dirichlet_ids = []
    if bcs:
        dirichlet_ids = ["on_boundary"]
        if hasattr(mesh, "extruded") and mesh.extruded:
            dirichlet_ids.extend(["bottom", "top"])

    tdim = mesh.topological_dimension
    if family.endswith("E"):
        family = "RTCE" if tdim == 2 else "NCE"
    if family.endswith("F"):
        family = "RTCF" if tdim == 2 else "NCF"

    fs = VectorFunctionSpace if vector else FunctionSpace

    V = fs(mesh, family, degree, variant=variant)
    v = TestFunction(V)
    u = TrialFunction(V)
    d = {
        H1: grad,
        HCurl: curl,
        HDiv: div,
    }[V.ufl_element().sobolev_space]

    formdegree = V.finat_element.formdegree
    if formdegree == 0:
        a = inner(d(u), d(v)) * dx
    else:
        a = (inner(u, v) + inner(d(u), d(v))) * dx

    u_exact = rg.uniform(V, -1, 1)
    L = ufl.replace(a, {u: u_exact})
    bcs = [DirichletBC(V, u_exact, sub) for sub in dirichlet_ids]
    nsp = None
    if formdegree == 0:
        b = np.zeros(V.value_shape)
        expr = Constant(b)
        basis = []
        for i in np.ndindex(V.value_shape):
            b[...] = 0
            b[i] = 1
            expr.assign(b)
            basis.append(Function(V).interpolate(expr))
        nsp = VectorSpaceBasis(basis)
        nsp.orthonormalize()

    uh = Function(V, name="solution")
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    sp = solver_parameters(condense, variant=variant)
    solver = LinearVariationalSolver(problem, near_nullspace=nsp,
                                     solver_parameters=sp)
    solver.solve()
    uerr = Function(V).assign(uh - u_exact)
    assert (assemble(a(uerr, uerr)) / assemble(a(u_exact, u_exact))) ** 0.5 < 1E-8
    return solver.snes.getLinearSolveIterations()


@pytest.fixture(params=(2, 3), ids=("square", "cube"))
def mh(request):
    dim = request.param
    nx = 3
    base = UnitSquareMesh(nx, nx, quadrilateral=True)
    mh = MeshHierarchy(base, 1)
    if dim == 3:
        mh = ExtrudedMeshHierarchy(mh, height=1, base_layer=nx)
    return mh


@pytest.mark.parallel
@pytest.mark.parametrize("degree", range(1, 3))
@pytest.mark.parametrize("variant", ("spectral", "fdm"))
def test_vertex_dofs(mh, variant, degree):
    """Check that we extract the right number of vertex dofs from a high order Lagrange space."""
    from firedrake.preconditioners.bddc import get_restricted_dofs
    mesh = mh[-1]
    P1 = FunctionSpace(mesh, "Lagrange", 1, variant=variant)
    V0 = FunctionSpace(mesh, "Lagrange", degree, variant=variant)
    v = get_restricted_dofs(V0, "vertex")
    assert v.getSizes() == P1.dof_dset.layout_vec.getSizes()


@pytest.mark.parallel
@pytest.mark.parametrize("family,degree", [("Q", 4)])
@pytest.mark.parametrize("condense", (False, True))
def test_bddc_fdm(rg, mh, family, degree, condense):
    """Test h-independence of condition number by measuring iteration counts"""
    variant = "fdm"
    bcs = True
    its = [solve_riesz_map(rg, m, family, degree, variant, bcs, condense=condense) for m in mh]
    assert (np.diff(its) <= 2).all()


@pytest.mark.parallel
@pytest.mark.parametrize("family,degree", [("Q", 4)])
@pytest.mark.parametrize("vector", (False, True), ids=("scalar", "vector"))
def test_bddc_aij_quad(rg, mh, family, degree, vector):
    """Test h-independence of condition number by measuring iteration counts"""
    variant = None
    bcs = True
    its = [solve_riesz_map(rg, m, family, degree, variant, bcs, vector=vector) for m in mh]
    assert (np.diff(its) <= 1).all()


@pytest.mark.parallel
@pytest.mark.parametrize("family,degree", [("CG", 3), ("N1curl", 3), ("N1div", 3)])
def test_bddc_aij_simplex(rg, family, degree):
    """Test h-independence of condition number by measuring iteration counts"""
    variant = None
    bcs = True
    meshes = [UnitCubeMesh(nx, nx, nx) for nx in (3, 6)]
    its = [solve_riesz_map(rg, m, family, degree, variant, bcs) for m in meshes]
    assert (np.diff(its) <= 1).all()
