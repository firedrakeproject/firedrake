import pytest
import numpy as np
from scipy.sparse import csr_matrix
import petsctools
from firedrake import *
from firedrake.adjoint import *


def petsc2numpy_vec(petsc_vec):
    """Allgather a PETSc.Vec."""
    gvec = petsc_vec
    gather, lvec = PETSc.Scatter().toAll(gvec)
    gather(gvec, lvec, addv=PETSc.InsertMode.INSERT_VALUES)
    return lvec.array_r.copy()


def petsc2numpy_mat(petsc_mat):
    """Allgather a PETSc.Mat."""
    comm = petsc_mat.getComm()
    local_mat = petsc_mat.getRedundantMatrix(
        comm.size, PETSc.COMM_SELF)
    return csr_matrix(
        local_mat.getValuesCSR()[::-1],
        shape=local_mat.getSize()
    ).todense()


@pytest.mark.skipcomplex
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("degree", (1, 2), ids=["degree1", "degree2"])
@pytest.mark.parametrize("dim", (0, 1, 2), ids=["scalar", "vec1", "vec2"])
@pytest.mark.parametrize("family", ("CG", "DG"))
@pytest.mark.parametrize("mesh_type", ("interval", "square"))
@pytest.mark.parametrize("backend", ("pyop2", "petsc"))
def test_white_noise(family, degree, mesh_type, dim, backend):
    """Test that white noise generator converges to a mass matrix covariance.
    """

    nx = 10
    # Mesh dimension
    if mesh_type == 'interval':
        mesh = UnitIntervalMesh(nx)
    elif mesh_type == 'square':
        mesh = UnitSquareMesh(nx, nx)
    elif mesh_type == 'cube':
        mesh = UnitCubeMesh(nx, nx, nx)

    # Variable rank
    if dim > 0:
        V = VectorFunctionSpace(mesh, family, degree, dim=dim)
    else:
        V = FunctionSpace(mesh, family, degree)

    # Finite element white noise has mass matrix covariance
    M = inner(TrialFunction(V), TestFunction(V))*dx
    covmat = petsc2numpy_mat(
        assemble(M, mat_type='aij').petscmat)

    rng = RandomGenerator(PCG64(seed=13))

    generator = WhiteNoiseGenerator(
        V, backend=WhiteNoiseGenerator.Backend(backend), rng=rng)

    # Test convergence as sample size increases
    nsamples = [50, 100, 200, 400, 800]

    samples = np.empty((V.dim(), nsamples[-1]))
    for i in range(nsamples[-1]):
        with generator.sample().dat.vec_ro as bv:
            samples[:, i] = petsc2numpy_vec(bv)

    covariances = [np.cov(samples[:, :ns]) for ns in nsamples]

    # Covariance matrix should converge at a rate of sqrt(n)
    errors = [np.linalg.norm(cov-covmat) for cov in covariances]
    normalised_errors = [err*sqrt(n) for err, n in zip(errors, nsamples)]
    normalised_errors /= normalised_errors[-1]

    # Loose tolerance because RNG
    tol = 0.2
    assert (1 - tol) < np.max(normalised_errors) < (1 + tol)


@pytest.mark.skipcomplex
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("m", (0, 2, 4))
@pytest.mark.parametrize("dim", (0, 1, 2), ids=["scalar", "vector1", "vector2"])
@pytest.mark.parametrize("family", ("CG", "DG"))
@pytest.mark.parametrize("mesh_type", ("interval", "square"))
def test_covariance_inverse_action(m, family, mesh_type, dim):
    """Test that covariance operator action and inverse are opposites.
    """

    nx = 20
    if mesh_type == 'interval':
        mesh = PeriodicUnitIntervalMesh(nx)
        x, = SpatialCoordinate(mesh)
        wexpr = cos(2*pi*x)
    elif mesh_type == 'square':
        mesh = PeriodicUnitSquareMesh(nx, nx)
        x, y = SpatialCoordinate(mesh)
        wexpr = cos(2*pi*x)*cos(4*pi*y)
    elif mesh_type == 'cube':
        mesh = PeriodicUnitCubeMesh(nx, nx, nx)
        x, y, z = SpatialCoordinate(mesh)
        wexpr = cos(2*pi*x)*cos(4*pi*y)*cos(pi*z)
    if dim > 0:
        V = VectorFunctionSpace(mesh, family, 1, dim=dim)
        wexpr = as_vector([-1**(j+1)*wexpr for j in range(dim)])
    else:
        V = FunctionSpace(mesh, family, 1)

    rng = WhiteNoiseGenerator(
        V, rng=RandomGenerator(PCG64(seed=13)))

    L = 0.1
    sigma = 0.9

    solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    if family == 'CG':
        form = AutoregressiveCovariance.DiffusionForm.CG
    elif family == 'DG':
        form = AutoregressiveCovariance.DiffusionForm.IP
    else:
        raise ValueError("Do not know which diffusion form to use for family {family}")

    B = AutoregressiveCovariance(
        V, L, sigma, m, rng=rng, form=form,
        solver_parameters=solver_parameters,
        options_prefix="")

    w = Function(V).project(wexpr)
    wcheck = B.apply_action(B.apply_inverse(w))

    tol = 1e-10

    assert errornorm(w, wcheck) < tol


@pytest.mark.skipcomplex
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("m", (0, 2, 4))
def test_covariance_inverse_action_hdiv(m):
    """Test that covariance operator action and inverse are opposites
    for hdiv spaces.
    """

    nx = 20
    mesh = PeriodicUnitSquareMesh(nx, nx)
    x, y = SpatialCoordinate(mesh)
    wexpr = cos(2*pi*x)*cos(4*pi*x)

    V = FunctionSpace(mesh, "BDM", 1)
    wexpr = as_vector([-1**(j+1)*wexpr for j in range(2)])

    L = 0.1
    sigma = 0.9

    solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    form = AutoregressiveCovariance.DiffusionForm.IP

    B = AutoregressiveCovariance(
        V, L, sigma, m, form=form,
        solver_parameters=solver_parameters,
        options_prefix="")

    w = Function(V).project(wexpr)
    wcheck = B.apply_action(B.apply_inverse(w))

    tol = 1e-8

    assert errornorm(w, wcheck) < tol


@pytest.mark.skipcomplex
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("m", (0, 2, 4))
@pytest.mark.parametrize("family", ("CG", "DG"))
def test_covariance_adjoint_norm(m, family):
    """Test that covariance operators are properly taped.
    """
    nx = 20
    L = 0.2
    sigma = 0.1

    mesh = UnitIntervalMesh(nx)
    x, = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, family, 1)

    u = Function(V).project(sin(2*pi*x))
    v = Function(V).project(2 - 0.5*sin(6*pi*x))

    if family == 'CG':
        form = AutoregressiveCovariance.DiffusionForm.CG
    elif family == 'DG':
        form = AutoregressiveCovariance.DiffusionForm.IP
    else:
        raise ValueError("Do not know which diffusion form to use for family {family}")

    B = AutoregressiveCovariance(V, L, sigma, m, form=form)

    continue_annotation()
    with set_working_tape() as tape:
        w = Function(V).project(u**4 + v)
        J = B.norm(w)
        Jhat = ReducedFunctional(J, Control(u), tape=tape)
    pause_annotation()

    m = Function(V).project(sin(2*pi*(x+0.2)))
    h = Function(V).project(sin(4*pi*(x-0.2)))

    taylor = taylor_to_dict(Jhat, m, h)

    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']


@pytest.mark.skipcomplex
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("m", (0, 2, 4))
@pytest.mark.parametrize("family", ("CG", "DG"))
@pytest.mark.parametrize("operation", ("action", "inverse"))
def test_covariance_mat(m, family, operation):
    """Test that covariance mat and pc apply correct and opposite actions.
    """
    nx = 20
    L = 0.2
    sigma = 0.9

    mesh = UnitIntervalMesh(nx)
    coords, = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, family, 1)

    if family == 'CG':
        form = AutoregressiveCovariance.DiffusionForm.CG
    elif family == 'DG':
        form = AutoregressiveCovariance.DiffusionForm.IP
    else:
        raise ValueError("Do not know which diffusion form to use for family {family}")

    B = AutoregressiveCovariance(V, L, sigma, m, form=form)

    operation = CovarianceMatCtx.Operation(operation)

    mat = CovarianceMat(B, operation=operation)

    expr = 2*pi*coords

    if operation == CovarianceMatCtx.Operation.ACTION:
        x = Function(V).project(expr).riesz_representation()
        y = Function(V)
        xcheck = x.copy(deepcopy=True)
        ycheck = y.copy(deepcopy=True)

        B.apply_action(xcheck, tensor=ycheck)

    elif operation == CovarianceMatCtx.Operation.INVERSE:
        x = Function(V).project(expr)
        y = Function(V.dual())
        xcheck = x.copy(deepcopy=True)
        ycheck = y.copy(deepcopy=True)

        B.apply_inverse(xcheck, tensor=ycheck)

    with x.dat.vec as xv, y.dat.vec as yv:
        mat.mult(xv, yv)

    # flip to primal space to calculate norms
    if operation == CovarianceMatCtx.Operation.INVERSE:
        y = y.riesz_representation()
        ycheck = ycheck.riesz_representation()

    assert errornorm(ycheck, y)/norm(ycheck) < 1e-12

    if operation == CovarianceMatCtx.Operation.INVERSE:
        y = y.riesz_representation()
        ycheck = ycheck.riesz_representation()

    ksp = PETSc.KSP().create()
    ksp.setOperators(mat)

    tol = 1e-8

    petsctools.set_from_options(
        ksp, options_prefix=str(operation),
        parameters={
            'ksp_monitor': None,
            'ksp_type': 'richardson',
            'ksp_max_it': 2,
            'ksp_rtol': tol,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.adjoint.CovariancePC',
        }
    )
    x.zero()

    with x.dat.vec as xv, y.dat.vec as yv:
        with petsctools.inserted_options(ksp):
            ksp.solve(yv, xv)

    # CovarianceOperator operations should
    # be exact inverses of each other.
    assert ksp.its == 1

    if operation == CovarianceMatCtx.Operation.ACTION:
        x = x.riesz_representation()
        xcheck = xcheck.riesz_representation()

    assert errornorm(xcheck, x)/norm(xcheck) < tol
