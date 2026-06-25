import pytest
from firedrake import *
import numpy as np

mg_params = {
    "pc_use_amat": False,
    "ksp_monitor": None,
    "ksp_type": "cg",
    "ksp_rtol": 1E-10,
    "ksp_atol": 0E-10,
    "ksp_norm_type": "natural",
    "pc_type": "mg",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "esteig_ksp_view_singularvalues": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star_construct_dim": 0,
        "pc_star_use_coloring": True,
        "pc_star_sub_sub_pc_type": "cholesky",
        "pc_star_sub_sub_pc_factor_mat_solver_type": "petsc",
    },
    "mg_coarse": {
        "ksp_type": "preonly",
        "pc_type": "redundant",
        "redundant_pc_type": "cholesky",
        "redundant_pc_factor_mat_solver_type": "mumps",
    }
}

fieldsplit_params = {
    "mat_type": "matfree",
    "pmat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "a11",
    # Build the block sweep from the true operator, but keep the
    # field-1 MG solve on the preconditioning block S via pc_use_amat=False.
    "pc_fieldsplit_diag_use_amat": False,
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "cholesky",
    "fieldsplit_1": mg_params,
}

scpc_params = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "mat_type": "matfree",
    "pc_python_type": "firedrake.SCPC",
    "pc_sc_eliminate_fields": "0",
    "condensed_field": mg_params,
}


def run_CG_problem(r, degree, quads=False, pc_type="scpc"):
    """
    Solves the Dirichlet problem for the elliptic equation:

    -div(grad(u)) = f in [0, 1]^2, u = g on the domain boundary.

    The source function f and g are chosen such that the analytic
    solution is:

    u(x, y) = sin(x*pi)*sin(y*pi).

    This test uses a CG discretization splitting interior and facet DOFs
    and Slate to perform the static condensation and local recovery.
    This solver uses multigrid on a mesh hierarchy to test coarsening of
    Slate objects.
    """

    # Set up problem domain
    mesh = UnitSquareMesh(2, 2, quadrilateral=quads)
    mh = MeshHierarchy(mesh, r-1)
    mesh = mh[-1]
    x = SpatialCoordinate(mesh)
    u_exact = sin(x[0]*pi)*sin(x[1]*pi)
    f = -div(grad(u_exact))

    # Set up function spaces
    e = FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degree)
    V = FunctionSpace(mesh, MixedElement(e["interior"], e["facet"]))
    uh = Function(V)

    # Formulate the CG method in UFL
    u = sum(TrialFunctions(V))
    v = sum(TestFunctions(V))
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx
    bcs = DirichletBC(V.sub(1), 0, "on_boundary")

    aP = None
    if pc_type == "scpc":
        params = scpc_params
    elif pc_type == "fieldsplit":
        params = fieldsplit_params
        ui, uf = TrialFunctions(V)
        vi, vf = TestFunctions(V)
        A = Tensor(a)
        AII = Block(A, (0, 0))
        AFI = Block(Tensor(inner(grad(ui), grad(vf))*dx), ((0, 1), 0))
        AIF = Block(Tensor(inner(grad(uf), grad(vi))*dx), (0, (0, 1)))
        aP = A - AFI * Inverse(AII) * AIF
    else:
        raise ValueError(f"Unrecognised pc_type {pc_type}")

    problem = LinearVariationalProblem(a, L, uh, bcs=bcs, aP=aP)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.snes.ksp.setErrorIfNotConverged(True)
    solver.solve()

    ksp = solver.snes.ksp
    if pc_type == "scpc":
        ksp = ksp.pc.getPythonContext().condensed_ksp
    else:
        ksp = ksp.pc.getFieldSplitSubKSP()[1]

    its = ksp.getIterationNumber()
    error = norm(u_exact-sum(uh), norm_type="L2")
    return error, its


@pytest.mark.parallel
@pytest.mark.parametrize(('degree', 'quads', 'rate'),
                         [(3, False, 3.75),
                          (5, True, 5.75)])
def test_cg_convergence(degree, quads, rate):
    errors = []
    for r in range(2, 5):
        error, its = run_CG_problem(r, degree, quads)
        errors.append(error)
        assert its <= 13

    diff = np.array(errors)
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > rate).all()


@pytest.mark.parametrize("refine", (2, 4))
def test_Jp_fieldsplit_mg(refine):
    degree = 3
    quad = False
    error_sc, its_sc = run_CG_problem(refine, degree, quad, pc_type="scpc")
    error_fs, its_fs = run_CG_problem(refine, degree, quad, pc_type="fieldsplit")
    assert its_sc == its_fs
