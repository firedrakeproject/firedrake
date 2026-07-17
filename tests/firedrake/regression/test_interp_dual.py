import pytest
import numpy as np
from firedrake import *
from firedrake.utils import complex_mode
from firedrake.matrix import MatrixBase
import ufl


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


@pytest.fixture
def M(fs):
    uhat = TrialFunction(fs)
    v = TestFunction(fs)
    return inner(uhat, v) * dx


@pytest.fixture(scope="module")
def V1(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture(scope="module")
def V2(mesh):
    return FunctionSpace(mesh, "CG", 2)


@pytest.fixture(scope="module")
def f1(mesh, V1):
    x, y = SpatialCoordinate(mesh)
    expr = cos(2*pi*x)*sin(2*pi*y)
    return Function(V1).interpolate(expr)


def test_interp_self(V1):
    a = assemble(conj(TestFunction(V1)) * dx)
    b = assemble(conj(TestFunction(V1)) * dx)
    a.interpolate(a)
    assert np.allclose(a.dat.data_ro, b.dat.data_ro)


def test_assemble_interp_adjoint_tensor(mesh, V1, f1):
    a = assemble(conj(TestFunction(V1)) * dx)
    assemble(interpolate(f1 * TestFunction(V1), a), tensor=a)

    x, y = SpatialCoordinate(mesh)
    f2 = Function(V1, name="f2").interpolate(
        exp(x) * y)

    assert np.allclose(assemble(a(f2)), assemble(Function(V1).interpolate(conj(f1 * f2)) * dx))


def test_assemble_interp_operator(V2, f1):
    # Check type
    If1 = interpolate(f1, V2)
    assert isinstance(If1, ufl.Interpolate)

    # -- I(f1, V2) -- #
    a = assemble(If1)
    b = assemble(interpolate(f1, V2))
    assert np.allclose(a.dat.data, b.dat.data)


def test_assemble_interp_matrix(V1, V2, f1):
    # -- I(v1, V2) -- #
    v1 = TrialFunction(V1)
    Iv1 = interpolate(v1, V2)
    assert Iv1.arguments()[0].function_space() == V2.dual()
    assert Iv1.arguments()[1].function_space() == V1

    b = assemble(interpolate(f1, V2))
    assert b.function_space() == V2

    # Get the interpolation matrix
    a = assemble(Iv1)
    assert a.arguments()[0].function_space() == V2.dual()
    assert a.arguments()[1].function_space() == V1
    assert a.petscmat.getSize() == (V2.dim(), V1.dim())

    # Check that `I * f1 == b` with I the interpolation matrix
    # and b the interpolation of f1 into V2.
    res = assemble(action(a, f1))
    assert res.function_space() == V2
    assert np.allclose(res.dat.data, b.dat.data)


def test_assemble_interp_tlm(V1, V2, f1):
    # -- Action(I(v1, V2), f1) -- #
    v1 = TrialFunction(V1)
    Iv1 = interpolate(v1, V2)
    b = assemble(interpolate(f1, V2))

    assembled_action_Iv1 = assemble(action(Iv1, f1))
    assert np.allclose(assembled_action_Iv1.dat.data, b.dat.data)


def test_assemble_interp_adjoint_matrix(V1, V2):
    # -- Adjoint(I(v1, V2)) -- #
    v1 = TrialFunction(V1)
    Iv1 = interpolate(v1, V2)

    v2 = TestFunction(V2)
    c2 = assemble(conj(v2) * dx)
    # Interpolation from V2* to V1*
    c1 = Cofunction(V1.dual()).interpolate(c2)
    # Interpolation matrix (V2* -> V1*)
    adj_Iv1 = adjoint(Iv1)
    a = assemble(adj_Iv1)
    assert a.arguments() == adj_Iv1.arguments()
    assert a.petscmat.getSize() == (V1.dim(), V2.dim())

    res = Cofunction(V1.dual())
    with c2.dat.vec_ro as x, res.dat.vec_ro as y:
        a.petscmat.mult(x, y)
    assert np.allclose(res.dat.data, c1.dat.data)


def test_assemble_interp_adjoint_model(V1, V2):
    # -- Action(Adjoint(I(v1, v2)), fstar) -- #
    v1 = TrialFunction(V1)
    Iv1 = interpolate(v1, V2)

    fstar = Cofunction(V2.dual())
    v = Argument(V1, 0)
    Ivfstar = assemble(interpolate(v, fstar))
    # Action(Adjoint(I(v1, v2)), fstar) <=> I(v, fstar)
    res = assemble(action(adjoint(Iv1), fstar))
    assert np.allclose(res.dat.data, Ivfstar.dat.data)


def test_assemble_interp_adjoint_complex(mesh, V1, V2, f1):
    if complex_mode:
        f1 = Constant(3 - 5.j) * f1

    a = assemble(conj(TestFunction(V1)) * dx)
    b = assemble(interpolate(f1 * TestFunction(V2), a))
    x, y = SpatialCoordinate(mesh)
    f2 = Function(V2, name="f2").interpolate(
        exp(x) * y)

    assert np.allclose(assemble(b(f2)), assemble(Function(V1).interpolate(conj(f1 * f2)) * dx))


def test_assemble_interp_rank0(V1, V2, f1):
    # -- Interpolate(f1, u2) (rank 0) -- #
    v2 = TestFunction(V2)
    # Set the Cofunction u2
    u2 = assemble(conj(v2) * dx)
    # Interpolate(f1, u2) <=> Action(Interpolate(f1, V2), u2)
    # a is rank 0 so assembling it produces a scalar.
    a = assemble(interpolate(f1, u2))
    # Compute numerically Action(Interpolate(f1, V2), u2)
    b = assemble(interpolate(f1, V2))
    with b.dat.vec_ro as x, u2.dat.vec_ro as y:
        res = x.dot(y)
    assert np.abs(a - res) < 1e-9


def test_assemble_base_form_operator_expressions(mesh):
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)

    x, y = SpatialCoordinate(mesh)
    f1 = Function(V1).interpolate(cos(2*pi*x)*sin(2*pi*y))
    f2 = Function(V1).interpolate(sin(2*pi*y))
    f3 = Function(V1).interpolate(cos(2*pi*x))

    If1 = interpolate(f1, V2)
    If2 = interpolate(f2, V2)
    If3 = interpolate(f3, V2)

    # Sum of BaseFormOperators (1-form)
    res = assemble(If1 + If2 + If3)
    res2 = assemble(assemble(If1) + assemble(If2) + assemble(If3))
    assert np.allclose(res.dat.data, res2.dat.data)

    # Sum of BaseFormOperator and Coefficients (1-form)
    u = Function(V2).interpolate(x**2 + y**2)
    res = assemble(u + If1)
    res2 = assemble(assemble(If1) + assemble(u))
    assert np.allclose(res.dat.data, res2.dat.data)

    # Sum of BaseFormOperator (2-form)
    v1 = TrialFunction(V1)
    Iv1 = interpolate(v1, V2)
    Iv2 = interpolate(v1, V2)
    res = assemble(Iv1 + Iv2)
    mat_Iv1 = assemble(Iv1)
    mat_Iv2 = assemble(Iv2)
    assert np.allclose(mat_Iv1.petscmat[:, :] + mat_Iv2.petscmat[:, :], res.petscmat[:, :], rtol=1e-14)

    # Linear combination of BaseFormOperator (1-form)
    alpha = 0.5
    res = assemble(alpha * If1 + alpha**2 * If2 - alpha ** 3 * If3)
    a = assemble(If1)
    b = assemble(If2)
    c = assemble(If3)
    res2 = assemble(alpha * a + alpha**2 * b - alpha**3 * c)
    assert np.allclose(res2.dat.data, res.dat.data)

    # Linear combination of BaseFormOperator (2-form)
    res = assemble(alpha * Iv1 - alpha**2 * Iv2)
    assert np.allclose(alpha * mat_Iv1.petscmat[:, :] - alpha**2 * mat_Iv2.petscmat[:, :], res.petscmat[:, :], rtol=1e-14)


def test_check_identity(mesh):
    V2 = FunctionSpace(mesh, "CG", 2)
    V1 = FunctionSpace(mesh, "CG", 1)
    v2 = TestFunction(V2)
    v1 = TestFunction(V1)
    a = assemble(interpolate(v1, conj(v2)*dx))
    b = assemble(conj(v1)*dx)
    assert np.allclose(a.dat.data, b.dat.data)


def test_solve_interp_f(mesh):
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "DG", 0)
    x, y = SpatialCoordinate(mesh)

    # The space of interpolation (V2) is voluntarily chosen to be of poorer quality than V1.
    # The reasons is that the form is defined on V1 so if we interpolate
    # in a higher-order space we won't see the impact of the interpolation.
    w = TestFunction(V1)
    u = Function(V1)
    f1 = Function(V1).interpolate(cos(x)*sin(y))

    # -- Exact solution with a source term interpolated into DG0
    f2 = assemble(interpolate(f1, V2))
    F = inner(grad(u), grad(w))*dx + inner(u, w)*dx - inner(f2, w)*dx
    solve(F == 0, u)

    # -- Solution where the source term is interpolated via `ufl.Interpolate`
    u2 = Function(V1)
    If = interpolate(f1, V2)
    # This requires assembling If
    F2 = inner(grad(u2), grad(w))*dx + inner(u2, w)*dx - inner(If, w)*dx
    solve(F2 == 0, u2)
    assert np.allclose(u.dat.data, u2.dat.data)


def test_solve_interp_u(mesh):
    V1 = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    f = Function(V1).interpolate(cos(x)*sin(y))

    # -- Exact solution
    F = inner(grad(u), grad(w))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    # -- Non mat-free case not supported yet => Need to be able to get the Interpolation matrix -- #
    """
    # -- Solution where the source term is interpolated via `ufl.Interpolate`
    u2 = Function(V1)
    # Iu is the identity
    Iu = Interpolate(u2, V1)
    # This requires assembling the Jacobian of Iu
    F2 = inner(grad(u), grad(w))*dx + inner(Iu, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)
    """

    # -- Solution where u2 is interpolated via `ufl.Interpolate` (mat-free)
    u2 = Function(V1)
    # Iu is the identity
    Iu = interpolate(u2, V1)
    # This requires assembling the action the Jacobian of Iu
    F2 = inner(grad(u2), grad(w))*dx + inner(Iu, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2, solver_parameters={"mat_type": "matfree",
                                          "ksp_type": "cg",
                                          "pc_type": "none"})
    assert np.allclose(u.dat.data, u2.dat.data)

    # Same problem with grad(Iu) instead of grad(Iu)
    u2 = Function(V1)
    # Iu is the identity
    Iu = interpolate(u2, V1)
    # This requires assembling the action the Jacobian of Iu
    F2 = inner(grad(Iu), grad(w))*dx + inner(Iu, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2, solver_parameters={"mat_type": "matfree",
                                          "ksp_type": "cg",
                                          "pc_type": "none"})
    assert np.allclose(u.dat.data, u2.dat.data)


@pytest.fixture(params=[("DG", 1, "CG", 2),
                        ("DG", 0, "RT", 1),
                        ("CG", 1),
                        ("RT", 1)],
                ids=lambda x: "-".join(map(str, x)))
def target_space(mesh, request):
    spaces = []
    for i in range(0, len(request.param), 2):
        family, degree = request.param[i:i+2]
        spaces.append(FunctionSpace(mesh, family, degree))

    W = spaces[0] if len(spaces) == 1 else MixedFunctionSpace(spaces)
    return W


@pytest.fixture(params=["scalar", "vector", "mixed"])
def source_space(mesh, request):
    if request.param == "scalar":
        return FunctionSpace(mesh, "DG", 0)
    elif request.param == "vector":
        return VectorFunctionSpace(mesh, "DG", 0, dim=3)
    elif request.param == "mixed":
        P0 = FunctionSpace(mesh, "DG", 0)
        return P0 * P0 * P0
    else:
        raise ValueError("Unrecognized parameter")


def test_interp_dual_mixed(source_space, target_space):
    W = target_space
    w = TestFunction(W)

    V = source_space
    v = TestFunction(V)

    A_shape = V.value_shape + W.value_shape
    if A_shape == ():
        A = 1
    else:
        mn = V.value_size * W.value_size
        A = as_tensor(np.arange(1, 1+mn).reshape(A_shape))

    if V.value_shape == ():
        b = 1
    else:
        m = V.value_size
        b = as_tensor(np.arange(1, 1+m).reshape(V.value_shape))

    expr = dot(A, w) if V.value_shape == () else A * w

    F_target = inner(b, expr)*dx(degree=0)
    expected = assemble(F_target)

    F_source = inner(b, v)*dx
    I_source = interpolate(expr, F_source)

    c = Cofunction(W.dual())
    c.assign(99)
    for tensor in (None, c):
        result = assemble(I_source, tensor=tensor)
        assert result.function_space() == W.dual()
        if tensor:
            assert result is tensor
        for x, y, in zip(result.subfunctions, expected.subfunctions):
            assert np.allclose(x.dat.data_ro, y.dat.data_ro)


def test_assemble_action_adjoint(V1, V2):
    u = TrialFunction(V1)

    a = interpolate(u, V2)  # V1 x V2^* -> R, equiv. V1 -> V2
    assert a.arguments() == (TestFunction(V2.dual()), TrialFunction(V1))

    f_form = inner(1, TestFunction(V2)) * dx

    for f in (f_form, assemble(f_form)):
        expr = action(adjoint(assemble(a)), f)
        assert isinstance(expr, Action)
        res = assemble(expr)
        assert isinstance(res, Cofunction)
        assert res.function_space() == V1.dual()

        expr2 = action(f, a)  # This simplifies into an Interpolate
        assert isinstance(expr2, Interpolate)
        res2 = assemble(expr2)
        assert isinstance(res2, Cofunction)
        assert res2.function_space() == V1.dual()
        assert np.allclose(res.dat.data, res2.dat.data)

        A = assemble(a)
        assert isinstance(A, MatrixBase)

        # This doesn't explicitly assemble the adjoint of A, but uses multHermitian
        expr3 = action(f, A)
        assert isinstance(expr3, Action)
        res3 = assemble(expr3)
        assert isinstance(res3, Cofunction)
        assert res3.function_space() == V1.dual()
        assert np.allclose(res.dat.data, res3.dat.data)

        # This is simplified into action(f, A) to avoid explicit assembly of adjoint(A)
        expr4 = action(adjoint(A), f)
        assert isinstance(expr4, Action)
        res4 = assemble(expr4)
        assert isinstance(res4, Cofunction)
        assert res4.function_space() == V1.dual()
        assert np.allclose(res.dat.data, res4.dat.data)


def test_assemble_interp_vector_matrix():
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    W = VectorFunctionSpace(mesh, "DG", 1, dim=2)
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(as_vector((x + 2*y, 2*x - y)))

    operator = assemble(interpolate(TrialFunction(V), W))
    actual = assemble(action(operator, f))
    expected = assemble(interpolate(f, W))

    assert np.allclose(actual.dat.data_ro, expected.dat.data_ro)


def test_assemble_interp_mixed_vector_matrix():
    mesh = UnitSquareMesh(2, 2)
    X = VectorFunctionSpace(mesh, "CG", 2)
    Y = VectorFunctionSpace(mesh, "DG", 1)
    V = FunctionSpace(mesh, "CG", 1)
    Z = X * V
    W = Y * V
    x, y = SpatialCoordinate(mesh)
    f = Function(Z)
    f.sub(0).interpolate(as_vector((x + 2*y, 2*x - y)))
    f.sub(1).interpolate(x - y)

    operator = assemble(interpolate(TrialFunction(Z), W), mat_type="nest")
    actual = assemble(action(operator, f))
    expected = assemble(interpolate(f, W))

    for actual_subfunction, expected_subfunction in zip(
        actual.subfunctions, expected.subfunctions
    ):
        assert np.allclose(
            actual_subfunction.dat.data_ro,
            expected_subfunction.dat.data_ro,
        )


def test_interpolate_mixed_vector_in_bilinear_form():
    from firedrake.assemble import ExplicitMatrixAssembler, get_assembler

    mesh = UnitSquareMesh(2, 2)
    X = VectorFunctionSpace(mesh, "CG", 2)
    Y = VectorFunctionSpace(mesh, "DG", 1)
    V = FunctionSpace(mesh, "CG", 1)
    Z = X * V
    W = Y * V
    x, y = SpatialCoordinate(mesh)
    f = Function(Z)
    f.sub(0).interpolate(as_vector((x + 2*y, 2*x - y)))
    f.sub(1).interpolate(x - y)
    v = TestFunction(W)
    form = inner(interpolate(TrialFunction(Z), W), v) * dx

    assembler = get_assembler(form, mat_type="nest")
    assert isinstance(assembler, ExplicitMatrixAssembler)
    operator = assembler.assemble()
    actual = assemble(action(operator, f))
    interpolated = assemble(interpolate(f, W))
    expected = assemble(inner(interpolated, v) * dx)

    for actual_subfunction, expected_subfunction in zip(
        actual.subfunctions, expected.subfunctions
    ):
        assert np.allclose(
            actual_subfunction.dat.data_ro,
            expected_subfunction.dat.data_ro,
        )


@pytest.mark.parallel(2)
def test_interpolate_in_form_compiled_reuse():
    from firedrake.assemble import OneFormAssembler, get_assembler

    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "DG", 1)
    x, y = SpatialCoordinate(mesh)
    u = Function(V)
    v = TestFunction(W)
    interpolation = interpolate(u, W)
    form = inner(interpolation, v) * dx
    assembler = get_assembler(form)

    assert isinstance(assembler, OneFormAssembler)
    for scale in (1, 3):
        u.interpolate(scale * (x + y))
        actual = assembler.assemble()
        expected = assemble(inner(assemble(interpolation), v) * dx)
        assert np.allclose(actual.dat.data, expected.dat.data)


def test_interpolate_in_form_mapped_derivative():
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "RT", 1)
    x, y = SpatialCoordinate(mesh)
    u = Function(V).interpolate(as_vector((x**2 + y, x - y**2)))
    interpolation = interpolate(u, W)

    actual = assemble(inner(grad(interpolation), grad(interpolation)) * dx)
    interpolated = assemble(interpolation)
    expected = assemble(inner(grad(interpolated), grad(interpolated)) * dx)
    assert np.isclose(actual, expected)


def test_interpolate_in_bilinear_form():
    mesh = UnitIntervalMesh(3)
    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(W)
    operator = assemble(inner(interpolate(u, W), v) * dx)

    x, = SpatialCoordinate(mesh)
    f = Function(V).interpolate(x + 1)
    actual = assemble(action(operator, f))
    expected = assemble(inner(assemble(interpolate(f, W)), v) * dx)
    assert np.allclose(actual.dat.data, expected.dat.data)


def test_interpolate_in_interior_facet_form():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "DG", 1)
    x, y = SpatialCoordinate(mesh)
    u = Function(V).interpolate(x**2 + y)
    v = TestFunction(W)
    interpolation = interpolate(u, W)

    actual = assemble(jump(interpolation) * jump(v) * dS)
    interpolated = assemble(interpolation)
    expected = assemble(jump(interpolated) * jump(v) * dS)
    assert np.allclose(actual.dat.data, expected.dat.data)


def test_cross_mesh_interpolate_in_form_uses_base_form_assembler():
    from firedrake.assemble import BaseFormAssembler, get_assembler

    source_mesh = UnitSquareMesh(1, 1)
    target_mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(source_mesh, "CG", 1)
    W = FunctionSpace(target_mesh, "CG", 1)
    v = TestFunction(W)
    form = inner(interpolate(Function(V), W), v) * dx(domain=target_mesh)

    assert isinstance(get_assembler(form), BaseFormAssembler)
