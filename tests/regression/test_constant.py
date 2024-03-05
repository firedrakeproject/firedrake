from firedrake import *
from ufl.formatting.ufl2unicode import ufl2unicode
from ufl.classes import IntValue
import numpy as np
import pytest


def test_scalar_constant():
    for m in [UnitIntervalMesh(5), UnitSquareMesh(2, 2), UnitCubeMesh(2, 2, 2)]:
        c = Constant(1, domain=m)
        # Check that the constant has the correct dimension.
        assert c._ad_dim() == 1
        assert abs(assemble(c*dx(domain=m)) - 1.0) < 1e-10


def test_scalar_constant_assign():
    for m in [UnitIntervalMesh(5), UnitSquareMesh(2, 2), UnitCubeMesh(2, 2, 2)]:
        c = Constant(1, domain=m)
        assert abs(assemble(c*dx(domain=m)) - 1.0) < 1e-10
        c.assign(4)
        assert abs(assemble(c*dx(domain=m)) - 4.0) < 1e-10


@pytest.mark.parametrize(('init', 'new_vals'),
                         ((1, ([1, 1], "x", [[1, 1], [1, 1]])),
                          ([1, 1], ([1, "x"], "x", 1, [[1, 1], [1, 1]])),
                          ([[1], [1]], ([1, "x"], "x", 1, [[1, 1], [1, 1]]))))
def test_constant_assign_mismatch(init, new_vals):
    c = Constant(init)
    for v in new_vals:
        with pytest.raises((ValueError, TypeError)):
            c.assign(v)


def test_constant_cast_to_float():
    val = 10.0
    c = Constant(val)
    assert float(c) == val  # raises a warning about casting float to complex


@pytest.mark.skipreal
def test_constant_cast_to_complex():
    val = 10.0 + 10.0j
    c = Constant(val)
    assert complex(c) == val


def test_indexed_vector_constant_cast_to_float():
    val = [10.0, 20.0]
    c = Constant(val)
    # Check that the constant has the correct dimension.
    assert c._ad_dim() == len(val)
    for i in range(2):
        assert float(c[i]) == val[i]


def test_vector_constant_2d():
    m = UnitSquareMesh(1, 1)
    n = FacetNormal(m)

    c = Constant([1, -1], domain=m)
    # Mesh is:
    # ,---.
    # |\  |
    # | \ |
    # |  \|
    # `---'
    # Normal is in (1, 1) direction
    assert abs(assemble(dot(c('+'), n('+'))*dS)) < 1e-10
    assert abs(assemble(dot(c('-'), n('+'))*dS)) < 1e-10

    # Normal is in (-1, -1) direction
    assert abs(assemble(dot(c('+'), n('-'))*dS)) < 1e-10
    assert abs(assemble(dot(c('-'), n('-'))*dS)) < 1e-10

    c.assign([1, 1])
    assert abs(assemble(dot(c('+'), n('+'))*dS) - 2) < 1e-10
    assert abs(assemble(dot(c('-'), n('+'))*dS) - 2) < 1e-10

    # Normal is in (-1, -1) direction
    assert abs(assemble(dot(c('+'), n('-'))*dS) + 2) < 1e-10
    assert abs(assemble(dot(c('-'), n('-'))*dS) + 2) < 1e-10


def test_tensor_constant():
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "CG", 1)
    v = Function(V)
    v.assign(1.0)
    c = ((1., 0.), (0., 2.))
    sigma = Constant(c)
    # Check that the constant has the correct dimension.
    assert sigma._ad_dim() == len(c[0]) * len(c[1])
    val = assemble(inner(v, dot(sigma, v))*dx)

    assert abs(val-3.0) < 1.0e-10


def test_constant_scalar_assign_distributes():
    m = UnitSquareMesh(1, 1)
    V = VectorFunctionSpace(m, 'CG', 1)

    f = Function(V)

    c = Constant(11)

    f.assign(c)

    assert np.allclose(f.dat.data_ro, 11)


def test_constant_vector_assign_works():
    m = UnitSquareMesh(1, 1)
    V = VectorFunctionSpace(m, 'CG', 1)

    f = Function(V)

    c = Constant([10, 11])

    f.assign(c)

    assert np.allclose(f.dat.data_ro[:, 0], 10)
    assert np.allclose(f.dat.data_ro[:, 1], 11)


def test_constant_vector_assign_to_scalar_error():
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    f = Function(V)

    c = Constant([10, 11])

    with pytest.raises(ValueError):
        f.assign(c)


def test_constant_vector_assign_to_vector_mismatch_error():
    m = UnitSquareMesh(1, 1)
    V = VectorFunctionSpace(m, 'CG', 1)

    f = Function(V)
    a = [10, 11, 12]
    c = Constant(a)
    # Check that the constant has the correct dimension.
    assert c._ad_dim() == len(a)
    with pytest.raises(ValueError):
        f.assign(c)


def test_constant_assign_to_mixed():
    m = UnitSquareMesh(1, 1)
    V = VectorFunctionSpace(m, 'CG', 1)

    W = V*V

    c = Constant([10, 11])
    f = Function(W)
    f.sub(0).assign(c)
    f.sub(1).assign(c)

    for d in f.dat.data_ro:
        assert np.allclose(d[:, 0], 10)
        assert np.allclose(d[:, 1], 11)


def test_constant_multiplies_function():
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    u = Function(V)

    u.assign(10)
    f = Function(V)

    c = Constant(11)
    f.assign(u * c)

    assert np.allclose(f.dat.data_ro, 110)


def test_fresh_constant_hashes_different():
    c = Constant(1)
    d = Constant(1)

    assert hash(c) != hash(d)


def test_constants_are_renumbered_in_form_signature():
    mesh = UnitSquareMesh(1, 1)
    mesh.init()
    c = Constant(1)
    d = Constant(1)

    assert c.count() != d.count()
    assert (c*dx(domain=mesh)).signature() == (d*dx(domain=mesh)).signature()


def test_constant_names_are_not_used_in_generated_code():
    mesh = UnitIntervalMesh(1)
    c = Constant(1.0, name="()")
    # should run without error
    assemble(c * dx(mesh))


@pytest.mark.skipcomplex
def test_correct_constants_are_used_in_split_form():
    # see https://github.com/firedrakeproject/firedrake/issues/3091
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    W = V * R
    uh, vh = Function(W), TestFunction(W)
    uh.sub(0).assign(Constant(1.0))
    u, lmbda = split(uh)
    v, tau = split(vh)
    L = Constant(0.5) * v

    G = (
        Constant(1.0) * inner(grad(u), grad(v)) * dx
        + Constant(10.0) * u * u * v * dx
        - L * dx
    )
    const_1 = Constant(1.0)
    H = G + (exp(const_1 - lmbda) - const_1) * tau * dx

    bcs = [DirichletBC(W.sub(0), Constant(0.0), "on_boundary")]

    solve(H == 0, uh, bcs=bcs)
    u, lambda_ = uh.subfunctions
    assert np.allclose(lambda_.dat.data, 1)


def test_constant_subclasses_are_correctly_numbered():
    class CustomConstant(Constant):
        pass

    const1 = CustomConstant(1.0)
    const2 = Constant(1.0)
    const3 = CustomConstant(1.0)

    assert const2.count() == const1.count() + 1
    assert const3.count() == const1.count() + 2


def test_derivative_wrt_constant():
    mesh = UnitIntervalMesh(5)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    c = Constant(5)
    f = Function(V).assign(7)
    solution_a = Function(V)
    solution_b = Function(V)

    a = (c**2)*inner(u, v) * dx
    L = inner(f, v) * dx
    solve(a == L, solution_a)

    d = derivative(a, c, IntValue(1))
    solve(d == L, solution_b)

    assert np.allclose(solution_b.dat.data_ro, float(c)/2*solution_a.dat.data_ro)


def test_constant_ufl2unicode():
    mesh = UnitIntervalMesh(1)
    a = Constant(1.0, name="a")
    b = Constant(2.0, name="b")
    F = a * a * b * b * dx(mesh)
    _ = ufl2unicode(F)

    dFda = derivative(F, u=a)
    dFdb = derivative(F, u=b)
    _ = ufl2unicode(dFda)
    _ = ufl2unicode(dFdb)

    dFda_du = derivative(F, u=a, du=ufl.classes.IntValue(1))
    dFdb_du = derivative(F, u=b, du=ufl.classes.IntValue(1))
    _ = ufl2unicode(dFda_du)
    _ = ufl2unicode(dFdb_du)
