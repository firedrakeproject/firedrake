from firedrake import *
import numpy

sparams = {
    "mat_type": "nest",
    "snes_max_it": 1,
    "snes_convergence_test": "skip",
    "ksp_type": "cg",
    "pc_type": "mg",
    "mg_coarse": {
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "jacobi",
        "fieldsplit_1_pc_type": "none",
    },
    "mg_levels": {
        "ksp_max_it": 1,
        "ksp_type": "richardson",
        "ksp_richardson_scale": 0.5,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "jacobi",
        "fieldsplit_1_pc_type": "none",
    },
}


def test_inject_real_space():
    base = UnitIntervalMesh(1)
    mh = MeshHierarchy(base, 3)
    mesh = mh[-1]
    V = FunctionSpace(mesh, "DG", 0)
    R = FunctionSpace(mesh, "R", 0)
    Z = V * R

    def create_solver(Z):
        z = Function(Z)
        u1, u2 = split(z)
        v1, v2 = TestFunctions(Z)
        F = inner((u1 + 1)**3 - Constant(8), v1) * dx + inner(u2, v2) * dx
        problem = NonlinearVariationalProblem(F, z)
        return NonlinearVariationalSolver(problem, solver_parameters=sparams)

    # Create two solvers
    solver1 = create_solver(Z)
    solver2 = create_solver(Z)
    # First Newton step with solver1
    solver1.solve()
    # Copy solution from solver1 to solver2
    solver2._problem.u.assign(solver1._problem.u)
    # Second Newton step with solver1
    solver1.solve()
    # First Newton step with solver2
    solver2.solve()
    # Test that the matrices were assembled used the same injected solution
    for level in range(len(mh)):
        mat1 = solver1.snes.ksp.pc.getMGSmoother(level).pc.getOperators()[0]
        mat2 = solver2.snes.ksp.pc.getMGSmoother(level).pc.getOperators()[0]
        assert numpy.allclose(mat1.getNestSubMatrix(0, 0)[:, :], mat2.getNestSubMatrix(0, 0)[:, :])
