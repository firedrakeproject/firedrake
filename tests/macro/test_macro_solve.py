import numpy as np
import pytest
from firedrake import *


@pytest.fixture(params=("square", "cube"))
def mh(request):
    if request.param == "square":
        base_msh = UnitSquareMesh(2, 2)
    elif request.param == "cube":
        base_msh = UnitCubeMesh(2, 2, 2)
    return MeshHierarchy(base_msh, 3)


@pytest.fixture(params=["iso", "alfeld", "th"])
def variant(request):
    return request.param


@pytest.fixture
def mixed_element(mh, variant):
    cell = mh[0].ufl_cell()
    if variant == "iso":
        Vel = FiniteElement("CG", cell, degree=1, variant="iso")
        Pel = FiniteElement("CG", cell, degree=1)
    elif variant == "alfeld":
        Vel = FiniteElement("CG", cell, degree=2, variant="alfeld")
        Pel = FiniteElement("DG", cell, degree=1, variant="alfeld")
    elif variant == "th":
        Vel = FiniteElement("CG", cell, degree=2)
        Pel = FiniteElement("CG", cell, degree=1)
    return Vel, Pel


def conv_rates(x):
    x = np.asarray(x)
    return np.log2(x[:-1] / x[1:])


@pytest.fixture
def convergence_test(variant):
    if variant == "iso":
        def check(uerr, perr):
            u_conv = conv_rates(uerr)
            return (u_conv >= 1.9).all() and np.allclose(perr, 0, atol=1.e-7)
    elif variant == "alfeld":
        def check(uerr, perr):
            return (np.allclose(uerr, 0, atol=1.e-10)
                    and np.allclose(perr, 0, atol=1.e-7))
    elif variant == "th":
        def check(uerr, perr):
            return (np.allclose(uerr, 0, atol=1.e-10)
                    and np.allclose(perr, 0, atol=1.e-7))
    return check


@pytest.fixture
def div_test(variant):
    if variant in ["iso", "th"]:
        return lambda x: norm(div(x)) > 1.e-5
    elif variant == "alfeld":
        return lambda x: norm(div(x)) <= 1.e-10


def test_riesz(mh, variant, mixed_element, convergence_test):
    dim = mh[0].geometric_dimension()
    u_err = []
    p_err = []
    el1, el2 = mixed_element
    for msh in mh:
        x = SpatialCoordinate(msh)
        uexact = (sum(x),) + tuple(x[i]**2 for i in range(dim-1))
        zexact = (*uexact, x[dim-1])
        V = VectorFunctionSpace(msh, el1)
        Q = FunctionSpace(msh, el2)
        Z = V * Q
        u, p = TrialFunctions(Z)
        test = TestFunction(Z)
        v, q = split(test)

        a = inner(grad(u), grad(v)) * dx + inner(p, q) * dx
        L = a(test, as_vector(zexact))
        bcs = DirichletBC(Z[0], as_vector(zexact[:dim]), "on_boundary")

        zh = Function(Z)
        solve(a == L, zh, bcs=bcs)

        uh, ph = zh.subfunctions
        u_err.append(errornorm(as_vector(zexact[:dim]), uh))
        p_err.append(errornorm(zexact[-1], ph))

    assert convergence_test(u_err, p_err)


def stokes_mms(Z, zexact):
    trial = TrialFunction(Z)
    test = TestFunction(Z)
    u, p = split(trial)
    v, q = split(test)

    a = (inner(grad(u), grad(v)) * dx
         - inner(p, div(v)) * dx
         - inner(div(u), q) * dx)

    L = a(test, zexact)
    return a, L


def errornormL2_0(pexact, ph):
    msh = ph.function_space().mesh()
    vol = assemble(1*dx(domain=msh))
    return sqrt(abs(errornorm(pexact, ph)**2 - (1/vol)*assemble((pexact - ph)*dx)**2))


def test_stokes(mh, variant, mixed_element, convergence_test):
    dim = mh[0].geometric_dimension()
    sp = {"pc_factor_mat_ordering_type": "natural"} if variant == "th" or dim == 3 else None
    u_err = []
    p_err = []
    el1, el2 = mixed_element
    for msh in mh:
        x = SpatialCoordinate(msh)
        uexact = (sum(x),) + tuple(x[i]**2 for i in range(dim-1))
        zexact = (*uexact, x[dim-1])

        V = VectorFunctionSpace(msh, el1)
        Q = FunctionSpace(msh, el2)
        Z = V * Q

        a, L = stokes_mms(Z, as_vector(zexact))
        bcs = DirichletBC(Z[0], as_vector(zexact[:dim]), "on_boundary")

        nullspace = MixedVectorSpaceBasis(
            Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=Z.comm)])

        zh = Function(Z)
        solve(a == L, zh, bcs=bcs, nullspace=nullspace, solver_parameters=sp)
        uh, ph = zh.subfunctions
        u_err.append(errornorm(as_vector(zexact[:dim]), uh))
        p_err.append(errornormL2_0(zexact[-1], ph))

    assert convergence_test(u_err, p_err)


def test_div_free(mh, variant, mixed_element, div_test):
    dim = mh[0].geometric_dimension()
    el1, el2 = mixed_element
    for msh in mh:
        x = SpatialCoordinate(msh)
        V = VectorFunctionSpace(msh, el1)
        Q = FunctionSpace(msh, el2)
        Z = V * Q
        a, L = stokes_mms(Z, Constant([0] * (dim+1)))

        f = as_vector([1] + [0] * (dim-1))
        for k in range(1, dim):
            f = f * (x[k]*(1-x[k]))**2

        sub = tuple(range(1, 2*dim+1))
        bcs = [DirichletBC(Z[0], f, sub)]
        zh = Function(Z)
        solve(a == L, zh, bcs=bcs)
        uh, _ = zh.subfunctions
        assert div_test(uh)
