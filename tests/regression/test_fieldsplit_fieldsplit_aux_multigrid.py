"""
This test segfaults without commit 5a018d27601ff5b06749abd94829c40a07b73877.

It seems like all the components are necessary, in particular
the boundary conditions, nonlinear problem, and the initial guess.

The segfault seems to occur when attempting multigrid on an
AuxiliaryOperatorPC approximating a Schur complement within
a fieldsplit within another fieldsplit.
"""
import pytest
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


def BoundaryConditions(mesh):
    y_component = Constant(0)
    x_component = Constant(1)
    return as_vector([x_component, y_component])


class SchurApprox(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        def alpha(d):
            alphabar = 2.5e4
            q = 0.1
            return alphabar * q * ((q + 1)/(d + q) - 1)

        ctx = self.get_appctx(pc)
        d = split(ctx["state"])[0]
        (u, p) = split(trial)
        (v, q) = split(test)
        K = (alpha(d) * inner(u, v)*dx
             + inner(grad(u), grad(v))*dx
             - inner(p, div(v))*dx
             - inner(q, div(u))*dx)
        expr = BoundaryConditions(test.function_space().mesh())
        bcs = [DirichletBC(test.function_space().sub(0), expr, "on_boundary")]
        return (K, bcs)


@pytest.mark.skipif(utils.complex_mode, reason="inner(grad(u), grad(u)) not complex Gateaux differentiable.")
def test_fieldsplit_fieldsplit_aux_multigrid():
    # Setup
    mesh = UnitSquareMesh(10, 10)
    hierarchy = MeshHierarchy(mesh, 1)
    mesh = hierarchy[-1]
    expr = BoundaryConditions(mesh)

    Ve = VectorElement("CG", triangle, 2, dim=2)  # Velocity
    Ce = FiniteElement("CG", triangle, 1)  # Density and pressure
    Re = FiniteElement("R", triangle, 0)   # Lagrange multiplier

    Ze = MixedElement([Ce, Ve, Ce, Re])
    Z = FunctionSpace(mesh, Ze)
    Ge = MixedElement([Ve, Ce])
    G = FunctionSpace(mesh, Ge)
    Gbcs = [DirichletBC(G.sub(0), expr, "on_boundary")]

    z = Function(Z)
    w = TestFunction(Z)

    d, u, p, r = split(z)

    def alpha(d):
        alphabar = 2.5e4
        q = 0.1
        return alphabar * q * ((q + 1)/(d + q) - 1)

    J = (0.5*inner(grad(u), grad(u))*dx
         - inner(p, div(u))*dx
         + 0.5*alpha(d)*inner(u, u)*dx
         - inner(r, 1./3.-d)*dx)

    F = derivative(J, z, w)
    bcs = [DirichletBC(Z.sub(1), expr, "on_boundary")]
    nsp = MixedVectorSpaceBasis(
        Z,
        [Z.sub(0), Z.sub(1), VectorSpaceBasis(
            constant=True, comm=mesh.comm
        ), Z.sub(3)]
    )

    # Initial guess
    gamma = 1./3.
    g = Function(G)
    (u_guess, p_guess) = split(g)
    J_guess = (0.5 * inner(grad(u_guess), grad(u_guess))*dx
               - inner(p_guess, div(u_guess))*dx
               + 0.5 * alpha(gamma) * inner(u_guess, u_guess)*dx)
    F_guess = derivative(J_guess, g, TestFunction(G))
    solver_params_guess = {"ksp_type": "preonly",
                           "mat_type": "aij",
                           "pc_type": "lu",
                           "pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER,
                           "snes_monitor": None}
    nsp_guess = MixedVectorSpaceBasis(
        G,
        [G.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm)]
    )
    solve(F_guess == 0, g, bcs=Gbcs, nullspace=nsp_guess, solver_parameters=solver_params_guess)
    (u_guess, p_guess) = g.subfunctions
    z.subfunctions[0].interpolate(Constant(gamma))
    z.subfunctions[1].assign(u_guess)
    z.subfunctions[2].assign(p_guess)
    z.subfunctions[3].interpolate(Constant(10))

    solver_params = {
        "snes_monitor": None,
        "ksp_type": "fgmres",
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "ksp_rtol": 1e-9,

        "mat_type": "matfree",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_0_fields": "0,1,2",
        "pc_fieldsplit_1_fields": "3",

        "fieldsplit_1_ksp_type": "gmres",
        "fieldsplit_1_ksp_max_it": 1,
        "fieldsplit_1_ksp_convergence_test": "skip",
        "fieldsplit_1_pc_type": "none",

        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "fieldsplit",
        "fieldsplit_0_pc_fieldsplit_type": "schur",
        "fieldsplit_0_pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0_pc_fieldsplit_0_fields": "0",
        "fieldsplit_0_pc_fieldsplit_1_fields": "1,2",

        "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_fieldsplit_0_pc_type": "python",
        "fieldsplit_0_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_fieldsplit_0_assembled_pc_type": "lu",
        "fieldsplit_0_fieldsplit_0_assembled_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER,

        "fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_0_fieldsplit_1_pc_type": "python",
        "fieldsplit_0_fieldsplit_1_pc_python_type": __name__ + ".SchurApprox",
        "fieldsplit_0_fieldsplit_1_aux_pc_type": "mg",
        "fieldsplit_0_fieldsplit_1_aux_pc_mg_type": "full",
        "fieldsplit_0_fieldsplit_1_aux_mg_coarse_pc_type": "python",
        "fieldsplit_0_fieldsplit_1_aux_mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_fieldsplit_1_aux_mg_coarse_assembled": {
            "mat_type": "aij",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER,
        },
        "fieldsplit_0_fieldsplit_1_aux_mg_levels_ksp_type": "chebyshev",
        "fieldsplit_0_fieldsplit_1_aux_mg_levels_ksp_convergence_test": "skip",
        "fieldsplit_0_fieldsplit_1_aux_mg_levels_ksp_max_it": 3,
        "fieldsplit_0_fieldsplit_1_aux_mg_levels_pc_type": "none",
    }

    solve(F == 0, z, bcs=bcs, nullspace=nsp, solver_parameters=solver_params)
