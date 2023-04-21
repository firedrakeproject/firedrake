import pytest
from firedrake import *


@pytest.mark.parametrize(['quadrilateral', 'ptype'],
                         [(q, p)
                          for q in [True, False]
                          for p in ["lu", "jacobi"]])
def test_facet_split(quadrilateral, ptype):
    if ptype == "lu":
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
    elif ptype == "jacobi":
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
                "fieldsplit_1_ksp_atol": 1E-10,
            },
        }

    r = 2
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
    assert sqrt(assemble(inner(uh - u_exact, uh - u_exact) * dx)) < 1E-10
