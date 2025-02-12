import numpy as np
import pytest
from firedrake import *


@pytest.fixture(params=("square", "cube"))
def mh(request):
    if request.param == "square":
        base_msh = UnitSquareMesh(3, 3)
    elif request.param == "cube":
        base_msh = UnitCubeMesh(2, 2, 2)
    return MeshHierarchy(base_msh, 2)


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
        dim = mh[0].topological_dimension()
        Vel = FiniteElement("CG", cell, degree=dim, variant="alfeld")
        Pel = FiniteElement("DG", cell, degree=dim-1, variant="alfeld")
    elif variant == "th":
        Vel = FiniteElement("CG", cell, degree=2)
        Pel = FiniteElement("CG", cell, degree=1)
    return Vel, Pel


def mesh_sizes(mh):
    mesh_size = []
    for msh in mh:
        DG0 = FunctionSpace(msh, "DG", 0)
        h = Function(DG0).interpolate(CellDiameter(msh))
        with h.dat.vec as hvec:
            _, maxh = hvec.max()
        mesh_size.append(maxh)
    return mesh_size


def conv_rates(x, h):
    x = np.asarray(x)
    h = np.asarray(h)
    return np.log2(x[:-1] / x[1:]) / np.log2(h[:-1] / h[1:])


def riesz_map(Z, gamma=None):
    v, q = TestFunctions(Z)
    u, p = TrialFunctions(Z)
    a = inner(grad(u), grad(v))*dx
    if gamma is not None:
        a += inner(div(u) * gamma, div(v))*dx
        a += inner(p / gamma, q)*dx
    else:
        a += inner(p, q) * dx
    return a


def test_riesz(mh, variant, mixed_element):
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

        a = riesz_map(Z)
        test, trial = a.arguments()
        L = a(test, as_vector(zexact))
        bcs = DirichletBC(Z[0], as_vector(zexact[:dim]), "on_boundary")

        zh = Function(Z)
        solve(a == L, zh, bcs=bcs)
        uh, ph = zh.subfunctions
        u_err.append(errornorm(as_vector(zexact[:dim]), uh))
        p_err.append(errornorm(zexact[-1], ph))

    # Here u is quadratic and p is linear,
    # all spaces should recover u and p exactly except iso
    if variant == "iso":
        h = mesh_sizes(mh)
        assert conv_rates(u_err, h)[-1] >= 1.9
        assert np.allclose(p_err, 0, atol=1E-10)
    elif variant == "alfeld":
        assert np.allclose(u_err, 0, atol=5E-9)
        assert np.allclose(p_err, 0, atol=5E-7)
    elif variant == "th":
        assert np.allclose(u_err, 0, atol=1E-10)
        assert np.allclose(p_err, 0, atol=1E-8)


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
    p0 = assemble((pexact - ph) * dx) / assemble(1*dx(domain=msh))
    return errornorm(pexact - Constant(p0), ph)


def test_stokes(mh, variant, mixed_element):
    dim = mh[0].geometric_dimension()
    u_err = []
    p_err = []
    div_err = []
    el1, el2 = mixed_element
    for msh in mh:
        # Construct a stream-function
        x = SpatialCoordinate(msh)
        psi0 = lambda i, j: x[i] * (1-x[i]) * x[j] * (1-x[j])
        if dim == 3:
            psi = as_vector([psi0(1, 2), psi0(2, 0), psi0(0, 1)])
        else:
            psi = psi0(0, 1)
        # Divergence-free exact solution
        uexact = curl(psi)
        pexact = x[dim-1] - Constant(0.5)
        zexact = (*uexact, pexact)

        V = VectorFunctionSpace(msh, el1)
        Q = FunctionSpace(msh, el2)
        Z = V * Q

        a, L = stokes_mms(Z, as_vector(zexact))
        bcs = DirichletBC(Z[0], as_vector(zexact[:dim]), "on_boundary")

        zh = Function(Z)
        nullspace = MixedVectorSpaceBasis(
            Z,
            [Z.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)]
        )
        solve(a == L, zh, bcs=bcs, nullspace=nullspace)

        uh, ph = zh.subfunctions
        u_err.append(errornorm(as_vector(uexact), uh))
        p_err.append(errornormL2_0(pexact, ph))
        div_err.append(sqrt(abs(assemble(inner(div(uh), div(uh))*dx))))

    # Here u is cubic and p is linear.
    # Only 3D alfeld can recover u and p exactly.
    # 2D and 3D alfeld should be divergence-free.
    h = mesh_sizes(mh)
    if variant == "iso":
        assert conv_rates(u_err, h)[-1] >= 1.9
        assert conv_rates(p_err, h)[-1] >= 0.9
        assert conv_rates(div_err, h)[-1] >= 0.9
    elif variant == "alfeld":
        if dim == 3:
            assert np.allclose(u_err, 0, atol=1E-9)
            assert np.allclose(p_err, 0, atol=1E-9)
        else:
            assert conv_rates(u_err, h)[-1] >= dim + 0.9
            assert conv_rates(p_err, h)[-1] >= dim-1 + 0.9
        # Test div-free
        assert np.allclose(div_err, 0, atol=1E-10)
    elif variant == "th":
        assert conv_rates(u_err, h)[-1] >= 2.9
        assert (conv_rates(p_err, h)[-1] >= 1.9
                or np.allclose(p_err, 0, atol=1E-10))
        assert conv_rates(div_err, h)[-1] >= 1.9
