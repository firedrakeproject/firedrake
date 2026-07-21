from firedrake import *
import pytest
import numpy as np


@pytest.fixture
def mesh(request):
    return UnitSquareMesh(5, 5)


def test_hand_specified_quadrature(mesh):
    V = FunctionSpace(mesh, 'CG', 2)
    v = TestFunction(V)

    a = conj(v) * dx

    a_q0 = assemble(a, form_compiler_parameters={'quadrature_degree': 0})
    a_q2 = assemble(a, form_compiler_parameters={'quadrature_degree': 2})

    assert not np.allclose(a_q0.dat.data, a_q2.dat.data)


def test_hand_specified_max_quadrature():
    mesh = UnitIntervalMesh(1)

    x, = SpatialCoordinate(mesh)
    a = (x**4)*dx

    # These should be the same because we only need degree=4 for exact integration.
    x4 = assemble(a)
    x4_maxquad5 = assemble(a, form_compiler_parameters={"max_quadrature_degree": 5})

    assert np.isclose(x4, x4_maxquad5)

    # These should be the same because degree=2 will limit the quadrature
    x4_quad2 = assemble(a, form_compiler_parameters={"quadrature_degree": 2})
    x4_maxquad2 = assemble(a, form_compiler_parameters={"max_quadrature_degree": 2})

    assert np.isclose(x4_quad2, x4_maxquad2)


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("mat_type", ["matfree", "aij"])
@pytest.mark.parametrize("family", ["Quadrature", "Boundary Quadrature"])
def test_quadrature_element(mesh, family, mat_type, diagonal):
    V = FunctionSpace(mesh, family, 2)
    v = TestFunction(V)
    u = TrialFunction(V)
    if family == "Boundary Quadrature":
        a = inner(u, v) * ds + inner(u('+'), v('+')) * dS
    else:
        a = inner(u, v) * dx

    assemble(a, mat_type=mat_type, diagonal=diagonal)


@pytest.mark.parametrize("family", ["DG", "CG"])
@pytest.mark.parametrize("cell", ["triangle", "tetrahedron"])
@pytest.mark.parametrize("degree", [1, 3])
def test_collapsed_quadrature_sum_factorisation(cell, degree, family):
    """``dx(scheme="collapsed")`` on a simplicial "DG"/variant="integral"
    (i.e. `finat.spectral.Legendre`) or "CG"/variant="integral" (i.e.
    `finat.spectral.IntegratedLegendre`, exercising the
    `FIAT.expansions.C0_basis` recombination) space must produce the same
    assembled residual and matrix as ``dx(scheme="canonical")``, even
    though it takes the sum-factorized (Duffy/lattice) tabulation path
    in ``tsfc.fem`` rather than the standard dense one.

    ``"canonical"`` is the same collapsed Gauss-Jacobi rule as
    ``"collapsed"`` (same points, same weights,
    `finat.quadrature.collapsed_gauss_jacobi_quadrature`), but as a plain
    `finat.point_set.PointSet` rather than a `CollapsedTensorProductPointSet`
    -- so `finat.duffy.DuffyElement._duffy_applies` is `False` and it always
    takes the dense FIAT tabulation path. Comparing against it (rather than
    the default scheme, a different quadrature rule entirely for these
    degrees) isolates any discrepancy to the sum-factorized tabulation
    itself, not to a difference in quadrature choice.
    """
    mesh = {"triangle": UnitSquareMesh(2, 2),
            "tetrahedron": UnitCubeMesh(1, 1, 1)}[cell]
    V = FunctionSpace(mesh, family, degree, variant="integral")
    u = TrialFunction(V)
    v = TestFunction(V)
    w = Function(V)
    w.dat.data[:] = np.random.default_rng(0).random(w.dat.data.shape)

    # translate_coefficient path (forward transform): residual with a
    # derivative, mixing both the coefficient and argument sum-factorized
    # tabulations.
    L = inner(grad(w), grad(v)) * dx(scheme="canonical")
    L_collapsed = inner(grad(w), grad(v)) * dx(scheme="collapsed")
    b = assemble(L)
    b_collapsed = assemble(L_collapsed)
    assert np.allclose(b.dat.data, b_collapsed.dat.data, rtol=1e-10, atol=1e-10)

    # translate_argument path (backward transform): mass matrix.
    a = inner(u, v) * dx(scheme="canonical")
    a_collapsed = inner(u, v) * dx(scheme="collapsed")
    M = assemble(a).M.values
    M_collapsed = assemble(a_collapsed).M.values
    assert np.allclose(M, M_collapsed, rtol=1e-10, atol=1e-10)
