import pytest
from firedrake import *
from firedrake.petsc import PETSc


def test_moore_spence():

    try:
        from slepc4py import SLEPc
    except ImportError:
        pytest.skip(reason="SLEPc unavailable, skipping eigenvalue test")

    msh = IntervalMesh(1000, 1)
    V = FunctionSpace(msh, "CG", 1)
    R = FunctionSpace(msh, "R", 0)

    # elastica residual
    def residual(theta, lmbda, ttheta):
        return inner(grad(theta), grad(ttheta))*dx - inner(lmbda**2*sin(theta), ttheta) * dx

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

    petsc_M = assemble(inner(TrialFunction(V), TestFunction(V))*dx, bcs=bcs).petscmat
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
    z.subfunctions[0].assign(th)
    z.subfunctions[1].assign(lm)
    z.subfunctions[2].assign(eigenmode)

    # Write Moore-Spence system of equations
    theta, lmbda, phi = split(z)
    ttheta, tlmbda, tphi = TestFunctions(Z)
    F1 = residual(theta, lmbda, ttheta)
    F2 = derivative(residual(theta, lmbda, tphi), z, as_vector([phi, 0, 0]))
    F3 = inner(dot(phi, phi) - 1, tlmbda)*dx

    F = F1 + F2 + F3

    bcs = [DirichletBC(Z.sub(0), 0.0, "on_boundary"), DirichletBC(Z.sub(2), 0.0, "on_boundary")]

    solve(F == 0, z, bcs=bcs)
    with z.sub(1).dat.vec_ro as x:
        param = x.norm()

    assert abs(param - pi) < 1.0e-4
