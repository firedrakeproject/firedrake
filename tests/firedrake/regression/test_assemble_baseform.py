import pytest
import numpy as np
from firedrake import *
from firedrake.assemble import get_assembler
from firedrake.utils import ScalarType
import ufl


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope='module', params=['cg1', 'vcg1', 'tcg1',
                                        'cg1cg1', 'cg1cg1[0]', 'cg1cg1[1]',
                                        'cg1vcg1[0]', 'cg1vcg1[1]',
                                        'cg1dg0', 'cg1dg0[0]', 'cg1dg0[1]',
                                        'cg2dg1', 'cg2dg1[0]', 'cg2dg1[1]'])
def fs(request, mesh):
    cg1 = FunctionSpace(mesh, "CG", 1)
    cg2 = FunctionSpace(mesh, "CG", 2)
    vcg1 = VectorFunctionSpace(mesh, "CG", 1)
    tcg1 = TensorFunctionSpace(mesh, "CG", 1)
    dg0 = FunctionSpace(mesh, "DG", 0)
    dg1 = FunctionSpace(mesh, "DG", 1)
    return {'cg1': cg1,
            'vcg1': vcg1,
            'tcg1': tcg1,
            'cg1cg1': cg1*cg1,
            'cg1cg1[0]': (cg1*cg1)[0],
            'cg1cg1[1]': (cg1*cg1)[1],
            'cg1vcg1': cg1*vcg1,
            'cg1vcg1[0]': (cg1*vcg1)[0],
            'cg1vcg1[1]': (cg1*vcg1)[1],
            'cg1dg0': cg1*dg0,
            'cg1dg0[0]': (cg1*dg0)[0],
            'cg1dg0[1]': (cg1*dg0)[1],
            'cg2dg1': cg2*dg1,
            'cg2dg1[0]': (cg2*dg1)[0],
            'cg2dg1[1]': (cg2*dg1)[1]}[request.param]


@pytest.fixture
def f(fs):
    f = Function(fs, name="f")
    x = SpatialCoordinate(fs.mesh())[0]
    f.interpolate(as_tensor(np.full(f.ufl_shape, x)))
    return f


@pytest.fixture
def one(fs):
    one = Function(fs, name="one")
    one.interpolate(Constant(np.ones(one.ufl_shape)))
    return one


@pytest.fixture
def M(fs):
    uhat = TrialFunction(fs)
    v = TestFunction(fs)
    return inner(uhat, v) * dx


@pytest.fixture
def a(fs, f):
    v = TestFunction(fs)
    return inner(f, v) * dx


def test_assemble_cofun(a):
    res = assemble(a)
    assert isinstance(res, Cofunction)


def test_assemble_matrix(M):
    res = assemble(M)
    assert isinstance(res, ufl.Matrix)


def test_assemble_adjoint(M):
    res = assemble(adjoint(M))
    assembledM = assemble(M)
    res2 = assemble(adjoint(assembledM))
    assert isinstance(res, ufl.Matrix)
    assert res.M.handle == res.petscmat
    assert np.allclose(res.M.handle[:, :], res2.M.handle[:, :], rtol=1e-14)


def test_assemble_action(M, f):
    res = assemble(action(M, f))
    assembledM = assemble(M)
    expr = action(assembledM, f)

    res2 = assemble(expr)
    assert isinstance(res2, Cofunction)
    assert isinstance(res, Cofunction)
    for f, f2 in zip(res.subfunctions, res2.subfunctions):
        assert abs(f.dat.data.sum() - f2.dat.data.sum()) < 1.0e-12
        if f.function_space().rank == 2:
            assert abs(f.dat.data.sum() - 0.5*sum(f.function_space().shape)) < 1.0e-12
        else:
            assert abs(f.dat.data.sum() - 0.5*f.function_space().value_size) < 1.0e-12

    out = assemble(expr, tensor=res2)
    assert out is res2
    for f, f2 in zip(res.subfunctions, res2.subfunctions):
        assert abs(f.dat.data.sum() - f2.dat.data.sum()) < 1.0e-12
        if f.function_space().rank == 2:
            assert abs(f.dat.data.sum() - 0.5*sum(f.function_space().shape)) < 1.0e-12
        else:
            assert abs(f.dat.data.sum() - 0.5*f.function_space().value_size) < 1.0e-12


@pytest.mark.parametrize("scale", ["float", "numpy", "ufl", "Constant", "Real"])
def test_scalar_formsum(f, scale):
    c = Cofunction(f.function_space().dual())
    c.assign(1)

    q = action(c, f)
    expected = 2 * assemble(q)

    s1 = 1.5
    s2 = 0.5
    if scale == "numpy":
        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
    elif scale == "ufl":
        s1 = as_ufl(s1)
        s2 = as_ufl(s2)
    elif scale == "Constant":
        s1 = Constant(s1)
        s2 = Constant(s2)
    elif scale == "Real":
        mesh = f.function_space().mesh()
        R = FunctionSpace(mesh.unique(), "R", 0)
        s1 = Function(R, val=s1)
        s2 = Function(R, val=s2)

    formsum = s1 * q + s2 * q
    assert isinstance(formsum, ufl.form.FormSum)
    res2 = assemble(formsum)
    assert res2 == expected

    mesh = f.function_space().mesh().unique()
    R = FunctionSpace(mesh, "R", 0)
    tensor = Cofunction(R.dual())

    result = assemble(formsum, tensor=tensor)
    assert result is tensor
    assert result.dat.data_ro.item() == expected


def test_vector_formsum(a):
    res = assemble(a)
    preassemble = assemble(a + a)
    formsum = res + a
    res2 = assemble(formsum)

    assert isinstance(formsum, ufl.FormSum)
    assert isinstance(res2, Cofunction)
    assert isinstance(preassemble, Cofunction)
    for f, f2 in zip(preassemble.subfunctions, res2.subfunctions):
        assert np.allclose(f.dat.data, f2.dat.data, atol=1e-12)

    out = assemble(formsum, tensor=res2)
    assert out is res2
    for f, f2 in zip(preassemble.subfunctions, res2.subfunctions):
        assert np.allclose(f.dat.data, f2.dat.data, atol=1e-12)


def test_matrix_formsum(M):
    res = assemble(M)
    sumfirst = assemble(M+M)
    formsum = res + M
    assert isinstance(formsum, ufl.form.FormSum)
    res2 = assemble(formsum)
    assert isinstance(res2, ufl.Matrix)
    assert np.allclose(sumfirst.petscmat[:, :], res2.petscmat[:, :], rtol=1E-14)

    out = assemble(formsum, tensor=res2)
    assert out is res2
    assert np.allclose(sumfirst.petscmat[:, :], res2.petscmat[:, :], rtol=1E-14)


def test_formsum_vector_self(a):
    operand = assemble(a)
    tensor = assemble(a)

    w = (42, 3.1416, 666)
    formsum = w[0] * tensor + w[1] * operand + w[2] * tensor
    assert isinstance(formsum, ufl.FormSum)

    result = assemble(formsum, tensor=tensor)
    assert result is tensor

    expected = assemble(Constant(sum(w)) * a)
    for f, f2 in zip(expected.subfunctions, result.subfunctions):
        assert np.allclose(f.dat.data, f2.dat.data, atol=1e-12)


def test_formsum_matrix_self(M):
    operand = assemble(M)
    tensor = assemble(M)

    w = (42, 3.1416, 666)
    formsum = w[0] * tensor + w[1] * operand + w[2] * tensor
    assert isinstance(formsum, ufl.FormSum)

    result = assemble(formsum, tensor=tensor)
    assert result is tensor

    expected = assemble(Constant(sum(w)) * M)
    assert np.allclose(expected.petscmat[:, :], result.petscmat[:, :], rtol=1E-14)


def test_zero_form(M, f, one):
    zero_form = assemble(action(action(M, f), one))
    assert isinstance(zero_form, ScalarType.type)
    assert abs(zero_form - 0.5 * np.prod(f.ufl_shape)) < 1.0e-12


def test_tensor_output(a, M):

    # 1-form tensor
    V = a.arguments()[0].function_space()
    tensor = Cofunction(V.dual())
    formsum = assemble(a) + a
    res = assemble(formsum, tensor=tensor)

    assert isinstance(formsum, ufl.form.FormSum)
    assert res is tensor

    # 2-form tensor
    tensor = get_assembler(M).allocate()
    formsum = assemble(M) + M
    res = assemble(formsum, tensor=tensor)

    assert isinstance(formsum, ufl.form.FormSum)
    assert res is tensor


def test_cofunction_assign(a, M, f):
    c1 = assemble(a)
    # Scale the action to obtain a different value than c1
    c2 = assemble(2 * action(M, f))
    assert isinstance(c1, Cofunction)
    assert isinstance(c2, Cofunction)

    # Assign Cofunction to Cofunction
    c1.assign(c2)
    for a, b in zip(c1.subfunctions, c2.subfunctions):
        assert np.allclose(a.dat.data, b.dat.data)

    # Assign BaseForm to Cofunction
    c1.assign(action(M, f))
    for a, b in zip(c1.subfunctions, c2.subfunctions):
        assert np.allclose(a.dat.data, 0.5 * b.dat.data)


def test_cofunction_action(a, f):
    zero_form_ref = assemble(ufl.action(a, f))
    v = assemble(a)

    zero_form = assemble(action(v, f))
    assert np.allclose(zero_form, zero_form_ref, rtol=1.0e-14)

    zero_form = assemble(0.5 * action(v, f))
    assert np.allclose(zero_form, 0.5 * zero_form_ref, rtol=1.0e-14)

    zero_form = assemble(0.5 * action(v, f) - 0.25 * action(v, f))
    assert np.allclose(zero_form, 0.25 * zero_form_ref, rtol=1.0e-14)


def test_cofunction_riesz_representation(a):
    # Get a Cofunction
    c = assemble(a)
    assert isinstance(c, Cofunction)

    V = c.function_space().dual()
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define Riesz maps
    riesz_maps = {'L2': inner(u, v) * dx,
                  'H1': (inner(u, v) + inner(grad(u), grad(v))) * dx,
                  'l2': None}

    # Check Riesz representation for each map
    for riesz_map, mass in riesz_maps.items():

        # Get Riesz representation of c
        r = c.riesz_representation(riesz_map=riesz_map)

        assert isinstance(r, Function)
        assert r.function_space() == V

        if mass:
            M = assemble(mass)
            Mr = Function(V)
            with r.dat.vec_ro as v_vec:
                with Mr.dat.vec as res_vec:
                    M.petscmat.mult(v_vec, res_vec)
        else:
            # l2 mass matrix is identity
            Mr = Function(V, val=r.dat)

        # Check residual
        for a, b in zip(Mr.subfunctions, c.subfunctions):
            assert np.allclose(a.dat.data, b.dat.data, rtol=1e-14)


def test_function_riesz_representation(f):
    # Get a Function
    assert isinstance(f, Function)

    V = f.function_space()
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define Riesz maps
    riesz_maps = {'L2': inner(u, v) * dx,
                  'H1': (inner(u, v) + inner(grad(u), grad(v))) * dx,
                  'l2': None}

    # Check Riesz representation for each map
    for riesz_map, mass in riesz_maps.items():

        # Get Riesz representation of f
        r = f.riesz_representation(riesz_map=riesz_map)

        assert isinstance(r, Cofunction)
        assert r.function_space() == V.dual()

        if mass:
            M = assemble(mass)
            Mf = Function(V)
            with f.dat.vec_ro as v_vec:
                with Mf.dat.vec as res_vec:
                    M.petscmat.mult(v_vec, res_vec)
        else:
            # l2 mass matrix is identity
            Mf = Cofunction(V.dual(), val=f.dat)

        # Check residual
        for a, b in zip(Mf.subfunctions, r.subfunctions):
            assert np.allclose(a.dat.data, b.dat.data, rtol=1e-14)


def helmholtz(r, quadrilateral=False, degree=2, mesh=None):
    # Create mesh and define function space
    if mesh is None:
        mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    lmbda = 1
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2))
    a = (inner(grad(u), grad(v)) + lmbda * inner(u, v)) * dx

    assembled_matrix = assemble(a)
    preassemble_action = assemble(action(a, f))
    postassemble_action = assemble(action(assembled_matrix, f))

    assert np.allclose(preassemble_action.M.values, postassemble_action.M.values, rtol=1e-14)


def test_assemble_baseform_return_tensor_if_given():
    mesh = UnitIntervalMesh(1)
    space = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    test = TestFunction(space)

    form = ufl.conj(test) * dx
    tensor = Cofunction(space.dual())

    b0 = assemble(form)
    b1 = assemble(b0, tensor=tensor)

    assert b0 is not b1
    assert b1 is tensor


def test_preprocess_form_rank2_simplification(f):
    """Check case (1) in BaseFormAssembler.preprocess_form."""
    from firedrake.assemble import BaseFormAssembler
    from ufl.algorithms import expand_indices, expand_derivatives

    V = f.function_space()
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * dx
    M = assemble(a)

    expr = action(action(M, M), f)
    expr = BaseFormAssembler.preprocess_base_form(expr)

    # Compare preprocessed form with expected result => this requires expanding indices!
    try:
        expected = action(M, action(M, f))
        expected = BaseFormAssembler.preprocess_base_form(expected)
        assert expand_indices(expr) == expand_indices(expected)
    except (KeyError, NotImplementedError):
        # Index expansion doesn't seem to play well with tensor elements.
        pass


def test_preprocess_form_adjoint_simplification(f):
    """Check cases (1), (2), (4), and (8) in BaseFormAssembler.preprocess_form."""
    from firedrake.assemble import BaseFormAssembler
    from ufl.algorithms import expand_indices, expand_derivatives

    V = f.function_space()

    try:
        # I(f, v*)
        I = interpolate(f, V)
    except NotImplementedError:
        # Interpolation is not implemented for certain spaces.
        I = f

    # N(I(f); v*)
    N = point_expr(lambda x: x, function_space=V)
    N = N(I)
    # Action(∂N(I(f); v1, v*)/∂I, ∂I/∂f) = Action(∂N(I(f); v1, v*)/∂I, I(v1, v*))
    dNdf = expand_derivatives(derivative(N, f))

    # F = <u, v> * dx
    u = TrialFunction(V)
    v = TestFunction(V)
    dFdN = inner(u, v) * dx

    # Action(Adjoint(Action(F, dNdf)), df)
    w = Function(V)
    expr = action(adjoint(action(dFdN, dNdf)), w)

    # BaseForm preprocessing
    expr = BaseFormAssembler.preprocess_base_form(expr)

    # Compare preprocessed form with expected result => this requires expanding indices!
    try:
        # Expanding indices is needed to match equal (different MultiIndex used for both).
        dF = action(adjoint(dFdN), w)
        dN = N._ufl_expr_reconstruct_(I, derivatives=(1,), argument_slots=(expand_indices(dF), v))
        # Reconstruct the expected preprocessed form
        expected = I._ufl_expr_reconstruct_(v, dN)

        # Expanding indices for the preprocessed form
        dN, v = expr.argument_slots()
        dF, w = dN.argument_slots()
        dN = dN._ufl_expr_reconstruct_(*dN.ufl_operands, argument_slots=(expand_indices(dF), w))
        expr = expr._ufl_expr_reconstruct_(v, dN)

        assert expr == expected
    except (KeyError, NotImplementedError):
        # Index expansion doesn't seem to play well with tensor elements.
        pass
