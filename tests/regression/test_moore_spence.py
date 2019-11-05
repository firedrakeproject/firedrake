import pytest
from firedrake import *
from firedrake.petsc import PETSc


@pytest.mark.parallel(nprocs=2)
def test_moore_spence():

    try:
        from slepc4py import SLEPc
    except ImportError:
        pytest.skip(msg="SLEPc unavailable, skipping eigenvalue test")

    msh = IntervalMesh(1000, 1)
    V = FunctionSpace(msh, "CG", 1)
    R = FunctionSpace(msh, "R", 0)

    # elastica residual
    def residual(theta, lmbda, ttheta):
        return inner(grad(theta), grad(ttheta))*dx - lmbda**2*sin(theta)*ttheta*dx

    th = Function(V)
    x = SpatialCoordinate(msh)[0]
    tth = TestFunction(V)
    lm = Constant(3.142)

    # Using guess for parameter lm, solve for state theta (th)
    A = residual(th, lm, tth)
    bcs = [DirichletBC(V, 0.0, "on_boundary")]
    solve(A == 0, th, bcs=bcs)

    # Now solve eigenvalue problem for $F_u(u, \lambda)\phi = r\phi$
    # Want eigenmode phi with minimal eigenvalue r
    B = derivative(residual(th, lm, TestFunction(V)), th, TrialFunction(V))

    petsc_M = assemble(inner(TestFunction(V), TrialFunction(V))*dx, bcs=bcs).petscmat
    petsc_B = assemble(B, bcs=bcs).petscmat

    num_eigenvalues = 1

    opts = PETSc.Options()
    opts.setValue("eps_target_magnitude", None)
    opts.setValue("eps_target", 0)
    opts.setValue("st_type", "sinvert")

    es = SLEPc.EPS().create(comm=COMM_WORLD)
    es.setDimensions(num_eigenvalues)
    es.setOperators(petsc_B, petsc_M)
    es.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    es.setFromOptions()
    es.solve()

    ev_re, ev_im = petsc_B.getVecs()
    es.getEigenpair(0, ev_re, ev_im)
    eigenmode = Function(V)
    eigenmode.vector().set_local(ev_re)

    Z = MixedFunctionSpace([V, R, V])

    # Set initial guesses for state, parameter, null eigenmode
    z = Function(Z)
    z.split()[0].assign(th)
    z.split()[1].assign(lm)
    z.split()[2].assign(eigenmode)

    # Write Moore-Spence system of equations
    theta, lmbda, phi = split(z)
    ttheta, tlmbda, tphi = TestFunctions(Z)
    F1 = residual(theta, lmbda, ttheta)
    F2 = derivative(residual(theta, lmbda, tphi), z, as_vector([phi, 0, 0]))
    F3 = (inner(phi, phi) - 1)*tlmbda*dx

    F = F1 + F2 + F3

    bcs = [DirichletBC(Z.sub(0), 0.0, "on_boundary"), DirichletBC(Z.sub(2), 0.0, "on_boundary")]

    # Need to fieldsplit onto the real variable as assembly doesn't work with R
    sp = {
        "mat_type": "matfree",
        "snes_type": "newtonls",
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_linesearch_type": "basic",
        "ksp_type": "fgmres",
        "ksp_monitor_true_residual": None,
        "ksp_max_it": 10,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_0_fields": "0,2",
        "pc_fieldsplit_1_fields": "1",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "python",
        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_assembled_pc_type": "lu",
        "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
        "fieldsplit_0_assembled_mat_mumps_icntl_14": 200,
        "mat_mumps_icntl_14": 200,
        "fieldsplit_1_ksp_type": "gmres",
        "fieldsplit_1_ksp_monitor_true_residual": None,
        "fieldsplit_1_ksp_max_it": 1,
        "fieldsplit_1_ksp_convergence_test": "skip",
        "fieldsplit_1_pc_type": "none",
    }

    solve(F == 0, z, bcs=bcs, solver_parameters=sp)
    with z.sub(1).dat.vec_ro as x:
        param = x.norm()

    assert abs(param - pi) < 1.0e-4
