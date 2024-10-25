from firedrake import *
from firedrake.preconditioners.fdm import tabulate_exterior_derivative
import pytest


@pytest.fixture(params=["tetrahedron", "hexahedron"])
def mesh_hierarchy(request):
    nx = 5
    nlevels = 2
    cell = request.param
    if cell == "tetrahedron":
        base = UnitCubeMesh(nx, nx, nx)
        return MeshHierarchy(base, nlevels)
    elif cell == "hexahedron":
        base = UnitSquareMesh(nx, nx, quadrilateral=True)
        basemh = MeshHierarchy(base, nlevels)
        return ExtrudedMeshHierarchy(basemh, height=1, base_layer=nx)


def run_riesz_map(V, mat_type, max_it):
    jacobi = {
        "mat_type": mat_type,
        "ksp_type": "preonly",
        "pc_type": "jacobi",
    }
    potential = jacobi
    if V.mesh().extruded and mat_type == "aij":
        # Test equivalence of Jacobi and ASM on edge/face-stars
        formdegree = V.finat_element.formdegree
        relax = {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMExtrudedStarPC",
            "pc_star_construct_dim": formdegree,
            "pc_star_sub_sub_ksp_type": "preonly",
            "pc_star_sub_sub_pc_type": "jacobi",
        }
    else:
        relax = jacobi

    coarse = {
        "ksp_type": "preonly",
        "pc_type": "cholesky",
    }
    if mat_type == "matfree":
        coarse = {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled": coarse,
        }
    parameters = {
        "mat_type": mat_type,
        "ksp_max_it": max_it,
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "ksp_monitor": None,
        "ksp_convergence_test": "skip",
        "pc_type": "mg",
        "mg_coarse": coarse,
        "mg_levels": {
            "ksp_type": "chebyshev",
            "ksp_chebyshev_esteig": "0.75,0.25,0,1",
            "pc_type": "python",
            "pc_python_type": "firedrake.HiptmairPC",
            "hiptmair_mg_levels": relax,
            "hiptmair_mg_coarse": potential,
        },
    }

    u_exact = Constant((1, 2, 4))
    f = u_exact

    uh = Function(V)
    u = TrialFunction(V)
    v = TestFunction(V)

    d = {HCurl: curl,
         HDiv: div}[V.ufl_element().sobolev_space]
    a = (inner(d(u), d(v)) + inner(u, v)) * dx(degree=2)
    L = inner(f, v) * dx(degree=2)

    bcs = [DirichletBC(V, u_exact, "on_boundary")]
    appctx = {"get_gradient": tabulate_exterior_derivative,
              "get_curl": tabulate_exterior_derivative}
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs, form_compiler_parameters={"mode": "vanilla"})
    solver = LinearVariationalSolver(problem, solver_parameters=parameters, appctx=appctx)
    solver.solve()
    return errornorm(u_exact, uh)


@pytest.mark.skipcomplexnoslate
@pytest.mark.parametrize("mat_type", ["aij", "matfree"])
def test_hiptmair_hcurl(mesh_hierarchy, mat_type):
    mesh = mesh_hierarchy[-1]
    if mesh.ufl_cell().is_simplex():
        family = "N1curl"
        max_it = 14
    else:
        family = "NCE"
        max_it = 5
    V = FunctionSpace(mesh, family, degree=1)
    assert run_riesz_map(V, mat_type, max_it) < 1E-6


@pytest.mark.skipcomplexnoslate
@pytest.mark.parametrize("mat_type", ["aij", "matfree"])
def test_hiptmair_hdiv(mesh_hierarchy, mat_type):
    mesh = mesh_hierarchy[-1]
    if mesh.ufl_cell().is_simplex():
        family = "N1div"
        max_it = 14
    else:
        family = "NCF"
        max_it = 7
    V = FunctionSpace(mesh, family, degree=1)
    assert run_riesz_map(V, mat_type, max_it) < 1E-6
