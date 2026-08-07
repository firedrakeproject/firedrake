from firedrake import *
from firedrake.utils import single_mode
import pytest


@pytest.mark.parametrize("named", [False, True], ids=["unnamed", "named"])
def test_fieldsplit_mg_options_prefix(named):
    if named:
        names = ("V1", "V2")
    else:
        names = (None, None)
    refine = 2
    base = UnitIntervalMesh(10)
    mh = MeshHierarchy(base, refine)
    mesh = mh[-1]
    V1 = FunctionSpace(mesh, 'CG', 1, name=names[0])
    V2 = FunctionSpace(mesh, 'CG', 2, name=names[1])
    W = MixedFunctionSpace([V1, V2])
    params0 = {
        "pc_type": "mg",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "mg_coarse_mat_type": "aij",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
    }
    params1 = {
        "pc_type": "mg",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "mg_levels_0_mat_type": "aij",
        "mg_levels_0_ksp_type": "preonly",
        "mg_levels_0_pc_type": "lu",
    }
    sp = {
        "mat_type": "matfree",
        "ksp_rtol": 1E-12,
        "ksp_max_it": 12,
        "ksp_type": "cg",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_ksp_type": "preonly",
        f"fieldsplit_{names[0] or 0}": params0,
        f"fieldsplit_{names[1] or 1}": params1,
    }

    x = SpatialCoordinate(mesh)
    z_exact = as_vector([x[0], x[0]**2])
    z = Function(W)
    w = TrialFunction(W)
    v = TestFunction(W)
    a = inner(grad(w), grad(v)) * dx
    L = a(v, z_exact)
    bcs = [DirichletBC(W.sub(i), z_exact[i], "on_boundary") for i in range(len(W))]
    problem = LinearVariationalProblem(a, L, z, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()
    assert errornorm(z_exact, z) / norm(z_exact) < (1e-6 if single_mode else 1E-12)

    assert solver.snes.ksp.pc.getOperators()[0].getType() == "python"
    fsplit = solver.snes.ksp.pc.getFieldSplitSubKSP()
    assert len(fsplit) == len(W)
    for ksp in fsplit:
        coarse = ksp.pc.getMGSmoother(0)
        assert coarse.pc.getType() == "lu"
        assert coarse.getOperators()[0].getType().endswith("aij")
        for l in range(1, len(mh)):
            level = ksp.pc.getMGSmoother(l)
            assert level.pc.getType() == "jacobi"
            assert level.getOperators()[0].getType() == "python"
