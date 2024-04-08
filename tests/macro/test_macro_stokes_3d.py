import numpy as np
import pytest
from firedrake import *


@pytest.fixture
def mh():
    base_msh = UnitCubeMesh(2, 2, 2)
    return MeshHierarchy(base_msh, 2)


@pytest.fixture(params=["iso", "alfeld", "th"])
def variant(request):
    return request.param


@pytest.fixture
def mixed_element(variant):
    if variant == "iso":
        Vel = FiniteElement("CG", tetrahedron, 1, variant="iso")
        Pel = FiniteElement("CG", tetrahedron, 1)
    elif variant == "alfeld":
        Vel = FiniteElement("CG", tetrahedron, 3, variant="alfeld")
        Pel = FiniteElement("DG", tetrahedron, 2, variant="alfeld")
    elif variant == "th":
        Vel = FiniteElement("CG", tetrahedron, 2)
        Pel = FiniteElement("CG", tetrahedron, 1)
    return Vel, Pel


def conv_rates(x):
    x = np.asarray(x)
    return np.log2(x[:-1] / x[1:])


@pytest.fixture
def convergence_test(variant):
    if variant == "iso":
        def check(uerr, perr):
            u_conv = conv_rates(uerr)
            p_conv = conv_rates(perr)
            return (u_conv >= 1.9).all() and (p_conv >= 1.8).all()
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
    sp = {"pc_factor_mat_ordering_type": "natural",
            "ksp_monitor": None,
            "ksp_type": "gmres",
            }# if variant == "th" else None
    u_err = []
    p_err = []
    el1, el2 = mixed_element
    for msh in mh:
        x, y, z = SpatialCoordinate(msh)
        zexact = [x+y, x**2, x-y**2, x+y+z]

        V = VectorFunctionSpace(msh, el1)
        Q = FunctionSpace(msh, el2)
        Z = V * Q

        a, L = stokes_mms(Z, as_vector(zexact))
        bcs = DirichletBC(Z[0], as_vector(zexact[:3]), "on_boundary")

        nullspace = MixedVectorSpaceBasis(
            Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
        nullspace = None

        zh = Function(Z)
        solve(a == L, zh, bcs=bcs, nullspace=nullspace, solver_parameters=sp)
        u_err.append(errornorm(as_vector(zexact[:3]), zh.subfunctions[0]))
        p_err.append(errornormL2_0(zexact[-1], zh.subfunctions[-1]))

    assert convergence_test(u_err, p_err)


def test_div_free(mh, variant, mixed_element, div_test):
    el1, el2 = mixed_element
    for msh in mh:
        x, y , z= SpatialCoordinate(msh)
        V = VectorFunctionSpace(msh, el1)
        W = FunctionSpace(msh, el2)
        Z = V * W
        a, L = stokes_mms(Z, Constant([0] * 4))
        bcs = [DirichletBC(Z[0], as_vector([y*(1-y)*z*(1-z), 0, 0]), (1, 3, 4, 5, 6))]

        zh = Function(Z)
        solve(a == L, zh, bcs=bcs)
        assert div_test(zh.subfunctions[0])
