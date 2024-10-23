import pytest
from firedrake import *


def run_facet_split(quadrilateral, pc_type, refine=2):
    if pc_type == "lu":
        parameters = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.FacetSplitPC",
            "facet": {
                "mat_type": "aij",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                "pc_fieldsplit_schur_precondition": "selfp",
                "fieldsplit_0_pc_type": "jacobi",
                "fieldsplit_1_pc_type": "lu",
            },
        }
    elif pc_type == "jacobi":
        parameters = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.FacetSplitPC",
            "facet": {
                "mat_type": "submatrix",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                "pc_fieldsplit_schur_precondition": "a11",
                "fieldsplit_0_pc_type": "jacobi",
                "fieldsplit_1_pc_type": "jacobi",
                "fieldsplit_1_ksp_type": "cg",
                "fieldsplit_1_ksp_rtol": 1E-12,
            },
        }

    r = refine
    variant = "fdm" if quadrilateral else None
    mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)

    V = FunctionSpace(mesh, FiniteElement("Lagrange", degree=3, variant=variant))
    u = TrialFunction(V)
    v = TestFunction(V)
    uh = Function(V)

    a = inner(grad(u), grad(v)) * dx
    L = inner(Constant(0), v) * dx
    x = SpatialCoordinate(mesh)
    u_exact = 42 * x[1]
    bcs = [DirichletBC(V, Constant(0), 3),
           DirichletBC(V, Constant(42), 4)]

    solve(a == L, uh, bcs=bcs, solver_parameters=parameters)
    return sqrt(assemble(inner(uh - u_exact, uh - u_exact) * dx))


@pytest.mark.parametrize("quadrilateral", [True, False])
@pytest.mark.parametrize("pc_type", ["lu", "jacobi"])
def test_facet_split(quadrilateral, pc_type):
    assert run_facet_split(quadrilateral, pc_type) < 1E-10


@pytest.mark.parallel
@pytest.mark.parametrize("pc_type", ["lu", "jacobi"])
def test_facet_split_parallel(pc_type):
    assert run_facet_split(True, pc_type, refine=3) < 1E-10
