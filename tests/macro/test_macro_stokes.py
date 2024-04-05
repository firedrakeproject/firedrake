import numpy as np
import pytest
from firedrake import *


@pytest.fixture
def mh():
    base_msh = UnitSquareMesh(2, 2)
    return MeshHierarchy(base_msh, 3)


@pytest.fixture(params=["iso", "alfeld", "th"])
def variant(request):
    return request.param


@pytest.fixture
def mixed_element(variant):
    if variant == "iso":
        Vel = FiniteElement("CG", triangle, 1, variant="iso")
        Pel = FiniteElement("CG", triangle, 1)
    elif variant == "alfeld":
        Vel = FiniteElement("CG", triangle, 2, variant="alfeld")
        Pel = FiniteElement("DG", triangle, 1, variant="alfeld")
    elif variant == "th":
        Vel = FiniteElement("CG", triangle, 2)
        Pel = FiniteElement("CG", triangle, 1)
    return Vel, Pel


def conv_rates(x):
    x = np.asarray(x)
    return np.log2(x[:-1] / x[1:])


@pytest.fixture
def convergence_test(variant):
    if variant == "iso":
        def check(uerr, perr):
            u_conv = conv_rates(uerr)
            return (u_conv >= 1.9).all() and np.allclose(perr, 0, rtol=1.e-10)
    elif variant == "alfeld":
        def check(uerr, perr):
            return (np.allclose(uerr, 0, rtol=1.e-10)
                    and np.allclose(perr, 0, rtol=1.e-10))
    elif variant == "th":
        def check(uerr, perr):
            return (np.allclose(uerr, 0, rtol=1.e-10)
                    and np.allclose(perr, 0, rtol=1.e-10))
    return check


@pytest.fixture
def div_test(variant):
    if variant in ["iso", "th"]:
        return lambda x: norm(div(x)) > 1.e-5
    elif variant == "alfeld":
        return lambda x: norm(div(x)) <= 1.e-10


def test_stokes(mh, variant, mixed_element, convergence_test):
    u_err = []
    p_err = []
    el1, el2 = mixed_element
    for msh in mh:
        x, y = SpatialCoordinate(msh)
        V = VectorFunctionSpace(msh, el1)
        W = FunctionSpace(msh, el2)
        Z = V * W
        u, p = TrialFunctions(Z)
        test = TestFunction(Z)
        v, w = split(test)

        upexact = [x+y, x**2, x]

        a = (inner(grad(u), grad(v)) * dx
             - inner(p, div(v)) * dx
             - inner(div(u), w) * dx)

        L = a(test, as_vector(upexact))
        bcs = DirichletBC(Z[0], as_vector(upexact[:2]), "on_boundary")

        nullspace = MixedVectorSpaceBasis(
            Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

        uph = Function(Z)

        solve(a == L, uph, bcs=bcs, nullspace=nullspace)

        u_err.append(errornorm(as_vector(upexact[:2]), uph.subfunctions[0]))
        p_err.append(errornorm(upexact[-1]-assemble(upexact[-1]*dx), uph.subfunctions[-1]))

    assert convergence_test(u_err, p_err)


def test_div_free(mh, variant, mixed_element, div_test):
    el1, el2 = mixed_element
    for msh in mh:
        x, y = SpatialCoordinate(msh)
        V = VectorFunctionSpace(msh, el1)
        W = FunctionSpace(msh, el2)
        Z = V * W
        u, p = TrialFunctions(Z)
        test = TestFunction(Z)
        v, w = split(test)

        a = (inner(grad(u), grad(v)) * dx
             - inner(p, div(v)) * dx
             - inner(div(u), w) * dx)
        L = inner(Constant((0, 0)), v) * dx
        bcs = [DirichletBC(Z[0], as_vector([y**2*(1-y)**2, 0]), 1),
               DirichletBC(Z[0], as_vector([y**2*(1-y)**2, 0]), (3, 4))]

        nullspace = MixedVectorSpaceBasis(
            Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

        uph = Function(Z)

        solve(a == L, uph, bcs=bcs, nullspace=nullspace)
        assert div_test(uph.subfunctions[0])
