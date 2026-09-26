import pytest
import numpy as np
from firedrake import *
from firedrake.utils import complex_mode
from firedrake.matrix import MatrixBase
from firedrake.assemble import BaseFormAssembler
from firedrake.petsc import PETSc
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


@pytest.mark.parallel([1, 3])
def test_solve_interp_hessian():
    # Fit a field to observations at a point cloud, regularised with a Laplacian term:
    #   J(u) = 1/2 sum_k |u(x_k) - p_k|^2 + alpha/2 |grad(u)|^2
    # The Hessian of the data term is I^T M I, with I the interpolation matrix from V
    # into the vertex-only mesh and M the mass matrix of the vertex-only mesh.
    mesh = UnitSquareMesh(5, 5)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    points = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.7], [0.3, 0.8]])
    vom = VertexOnlyMesh(mesh, points)
    P = FunctionSpace(vom, "DG", 0)
    p_obs = assemble(interpolate(1.0 + x + 2 * y, P))

    alpha = Constant(1e-2)
    u = Function(V)
    E = 0.5 * (interpolate(u, P) - p_obs)**2 * dx(vom)
    R = 0.5 * alpha * inner(grad(u), grad(u)) * dx
    J = E + R
    F = derivative(J, u)
    H = derivative(F, u)

    # -- Symbolic Hessian against explicit matrix products
    I = assemble(interpolate(TrialFunction(V), P)).petscmat
    M = assemble(inner(TrialFunction(P), TestFunction(P)) * dx).petscmat
    H_ref = assemble(derivative(derivative(R, u), u)).petscmat
    H_ref.axpy(1.0, M.ptap(I))
    H_diff = assemble(H).petscmat.copy()
    H_diff.axpy(-1.0, H_ref)
    assert H_diff.norm() < 1e-12 * H_ref.norm()

    # -- The nested derivative agrees with the derivative of the expanded residual
    F_expanded = BaseFormAssembler.preprocess_base_form(F)
    H_diff = assemble(derivative(F_expanded, u)).petscmat.copy()
    H_diff.axpy(-1.0, H_ref)
    assert H_diff.norm() < 1e-12 * H_ref.norm()

    # -- Solve the minimisation problem with the assembled Hessian
    solve(F == 0, u, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    with assemble(F).dat.vec_ro as residual:
        assert residual.norm() < 1e-10

    # -- Reference solve of (H_R + I^T M I) u = I^T p_obs with explicit matrices
    u_ref = Function(V)
    with u_ref.dat.vec as uv, p_obs.dat.vec_ro as pv:
        rhs = uv.duplicate()
        I.multTranspose(pv, rhs)
        ksp = PETSc.KSP().create(comm=mesh.comm)
        ksp.setOperators(H_ref)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(rhs, uv)
    assert np.allclose(u.dat.data_ro, u_ref.dat.data_ro)

    # -- The matrix-free Hessian applies the interpolation operator to a Function,
    # instead of assembling it into an intermediate matrix.
    w = Function(V).interpolate(cos(x) * sin(y))
    Hw_ref = Function(V.dual())
    with w.dat.vec_ro as wv, Hw_ref.dat.vec as rv:
        H_ref.mult(wv, rv)
    for expr in (H, BaseFormAssembler.preprocess_base_form(H)):
        H_matfree = assemble(expr, mat_type="matfree")
        Hw = Function(V.dual())
        with w.dat.vec_ro as wv, Hw.dat.vec as v:
            H_matfree.petscmat.mult(wv, v)
        assert np.allclose(Hw.dat.data_ro, Hw_ref.dat.data_ro)


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
