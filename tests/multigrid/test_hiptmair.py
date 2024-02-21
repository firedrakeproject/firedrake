from firedrake import *
from firedrake.preconditioners.fdm import tabulate_exterior_derivative
import pytest


def mesh_hierarchy(cell):
    nx = 5
    nlevels = 2
    if cell == "tetrahedron":
        base = UnitCubeMesh(nx, nx, nx)
        return MeshHierarchy(base, nlevels)
    elif cell == "hexahedron":
        base = UnitSquareMesh(nx, nx, quadrilateral=True)
        basemh = MeshHierarchy(base, nlevels)
        return ExtrudedMeshHierarchy(basemh, height=1, base_layer=nx)


def run_riesz_map(V, mat_type):
    relax = {
        "mat_type": mat_type,
        "ksp_type": "preonly",
        "pc_type": "jacobi",
    }
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
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "ksp_norm_type": "natural",
        "pc_type": "mg",
        "mg_coarse": coarse,
        "mg_levels": {
            "ksp_type": "chebyshev",
            "ksp_chebyshev_esteig": "0.75,0.25,0,1",
            "pc_type": "python",
            "pc_python_type": "firedrake.HiptmairPC",
            "hiptmair_mg_levels": relax,
            "hiptmair_mg_coarse": relax,
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
    if V.mesh().ufl_cell().is_simplex():
        appctx = dict()
    else:
        appctx = {"get_gradient": tabulate_exterior_derivative,
                  "get_curl": tabulate_exterior_derivative}
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs, form_compiler_parameters={"mode": "vanilla"})
    solver = LinearVariationalSolver(problem, solver_parameters=parameters, appctx=appctx)
    solver.solve()
    return solver.snes.ksp.getIterationNumber()


@pytest.mark.skipcomplexnoslate
@pytest.mark.parametrize(["family", "cell"],
                         [("N1curl", "tetrahedron"), ("NCE", "hexahedron")])
def test_hiptmair_hcurl(family, cell):
    mesh = mesh_hierarchy(cell)[-1]
    V = FunctionSpace(mesh, family, degree=1)
    assert run_riesz_map(V, "aij") <= 15
    assert run_riesz_map(V, "matfree") <= 15


@pytest.mark.skipcomplexnoslate
@pytest.mark.parametrize(["family", "cell"],
                         [("RT", "tetrahedron"), ("NCF", "hexahedron")])
def test_hiptmair_hdiv(family, cell):
    mesh = mesh_hierarchy(cell)[-1]
    V = FunctionSpace(mesh, family, degree=1)
    assert run_riesz_map(V, "aij") <= 12
    assert run_riesz_map(V, "matfree") <= 12
