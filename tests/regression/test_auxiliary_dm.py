from firedrake.petsc import *
from firedrake import *
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}


class ApproximateSchur(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        Z = test.function_space()
        (v, w) = split(trial)
        (y, z) = split(test)
        a = inner(div(v), div(y))*dx - inner(div(w), div(z))*dx + inner(v, z)*dx + inner(w, y)*dx
        bcs = [DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")]
        return (a, bcs)


class BiharmonicProblem(object):
    def __init__(self, baseN, nref):
        super().__init__()
        self.baseN = baseN
        self.nref = nref

    def mesh(self):
        base = RectangleMesh(self.baseN, self.baseN, 1, 1,
                             distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base, self.nref, distribution_parameters=distribution_parameters)
        mesh = mh[-1]
        return mesh

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.analytical_solution(Z.ufl_domain()), "on_boundary"),
               DirichletBC(Z.sub(1), Constant((0, 0)), "on_boundary")]
        return bcs

    def analytical_solution(self, mesh):
        # Define exact solution
        (x, y) = SpatialCoordinate(mesh)
        u_exact = (x*y*(1-x)*(1-y))**3
        return u_exact

    def src(self, mesh):
        # define RHS
        u_exact = self.analytical_solution(mesh)
        f = div(grad(div(grad(u_exact)))) + u_exact
        return f


def test_auxiliary_dm():
    problem = BiharmonicProblem(5, 1)
    mesh = problem.mesh()

    # Define function space
    Ue = FiniteElement("DG", mesh.ufl_cell(), 1)
    Ve = FiniteElement("RT", mesh.ufl_cell(), 2)
    Se = FiniteElement("RT", mesh.ufl_cell(), 2)

    Ze = MixedElement([Ue, Ve, Se])
    Z = FunctionSpace(mesh, Ze)

    # Define energy and corresponding weak form
    z = Function(Z)
    (u, v, alpha) = split(z)
    w = TestFunction(Z)

    f = problem.src(mesh)

    L = (
        + 0.5 * inner(div(v), div(v))*dx
        + 0.5 * inner(u, u)*dx
        + inner(alpha, v)*dx + inner(div(alpha), u)*dx
        - inner(f, u)*dx
        )  # noqa: E121

    F = derivative(L, z, w)

    # Impose Dirichlet BCs on u and v
    bcs = problem.bcs(Z)

    # Parameters for block factorization
    block_fact = {
                 "mat_type": "aij",        # noqa: E126
                 "snes_type": "newtonls",
                 "snes_atol": 2.0e-5,
                 "snes_rtol": 1.0e-5,
                 "snes_monitor": None,
                 "ksp_type": "fgmres",
                 "ksp_norm_type": "unpreconditioned",
                 "ksp_monitor": None,
                 "ksp_converged_reason": None,
                 "ksp_atol": 1e-10,
                 "ksp_rtol": 1e-10,
                 "ksp_max_it": 100,
                 "pc_type": "fieldsplit",
                 "pc_fieldsplit_0_fields": "0",
                 "pc_fieldsplit_1_fields": "1,2",
                 "pc_fieldsplit_type": "schur",
                 "pc_fieldsplit_schur_fact_type": "full",
                 "pc_fieldsplit_schur_precondition": "user",
                 "fieldsplit_0_ksp_type": "richardson",
                 "fieldsplit_0_pc_type": "ilu",
                 "fieldsplit_0_ksp_max_it": 1,
                 "fieldsplit_0_ksp_convergence_test": "skip",
                 "fieldsplit_1_ksp_type": "gmres",
                 "fieldsplit_1_ksp_max_it": 1,
                 "fieldsplit_1_ksp_convergence_test": "skip",
                 "fieldsplit_1_ksp_monitor_true_residual": None,
                 "fieldsplit_1_pc_type": "python",
                 "fieldsplit_1_pc_python_type": "test_auxiliary_dm.ApproximateSchur",
                 "fieldsplit_1_aux_pc_type": "mg",
                 "fieldsplit_1_aux_mg_levels_ksp_type": "richardson",
                 "fieldsplit_1_aux_mg_levels_ksp_richardson_scale": 1/3,
                 "fieldsplit_1_aux_mg_levels_pc_type": "python",
                 "fieldsplit_1_aux_mg_levels_pc_python_type": "firedrake.PatchPC",
                 "fieldsplit_1_aux_mg_levels_patch_pc_patch_save_operators": True,
                 "fieldsplit_1_aux_mg_levels_patch_pc_patch_partition_of_unity": False,
                 "fieldsplit_1_aux_mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
                 "fieldsplit_1_aux_mg_levels_patch_pc_patch_construct_dim": 0,
                 "fieldsplit_1_aux_mg_levels_patch_pc_patch_construct_type": "star",
                 "fieldsplit_1_aux_mg_levels_patch_sub_ksp_type": "preonly",
                 "fieldsplit_1_aux_mg_levels_patch_sub_pc_type": "svd",
                 "fieldsplit_1_aux_mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
                 "fieldsplit_1_aux_mg_coarse_pc_type": "lu",
                 "fieldsplit_1_aux_mg_coarse_pc_factor_mat_solver_type": "mumps",
                 "fieldsplit_1_aux_mg_coarse_mat_mumps_icntl_14": 200,
                 }

    # Solve variational form
    nvproblem = NonlinearVariationalProblem(F, z, bcs=bcs)
    prolongation = EmbeddedDGTransfer(Z.ufl_element())
    solver = NonlinearVariationalSolver(nvproblem, solver_parameters=block_fact)
    solver.set_transfer_operators(dmhooks.transfer_operators(Z, prolong=prolongation.prolong, inject=prolongation.inject, restrict=prolongation.restrict))
    solver.solve()

    # Error in L2 norm
    (u, v, alpha) = z.split()
    u_exact = problem.analytical_solution(mesh)
    error_L2 = errornorm(u_exact, u, 'L2') / errornorm(u_exact, Function(FunctionSpace(mesh, 'CG', 1)), 'L2')
    assert error_L2 < 0.02
