import pytest
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


class PointexprActionOperator(PointexprOperator):

    #def __init__(self, *args, **kwargs):
        #PointexprOperator.__init__(self, *args, **kwargs)
    def __init__(self, *operands, function_space, derivatives=None, result_coefficient=None, argument_slots=(),
                 val=None, name=None, dtype=ScalarType, operator_data):

        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                          result_coefficient=result_coefficient, argument_slots=argument_slots,
                                          val=val, name=name, dtype=dtype,
                                          operator_data=operator_data)

        # Check
        if not isinstance(operator_data, types.FunctionType):
            error("Expecting a FunctionType pointwise expression")
        expr_shape = operator_data(*operands).ufl_shape
        if expr_shape != function_space.ufl_element().value_shape():
            error("The dimension does not match with the dimension of the function space %s" % function_space)

    def _evaluate_action(self, args):
        if len(args) == 0:
            # Evaluate the operator
            return self._evaluate()

        # Evaluate the Jacobian/Hessian action
        operands = self.ufl_operands
        operator = self._compute_derivatives()
        expr = as_ufl(operator(*operands))
        if expr.ufl_shape == () and expr != 0:
            var = VariableRuleset(self.ufl_operands[0])
            expr = expr*var._Id
        elif expr == 0:
            return self.assign(expr)

        for arg in args:
            mi = indices(len(expr.ufl_shape))
            aa = mi
            bb = mi[-len(arg.ufl_shape):]
            expr = arg[bb] * expr[aa]
            mi_tensor = tuple(e for e in mi if not (e in aa and e in bb))
            if len(expr.ufl_free_indices):
                expr = as_tensor(expr, mi_tensor)
        return self.interpolate(expr)


def action_point_expr(point_expr, function_space):
    return partial(PointexprActionOperator, operator_data=point_expr, function_space=function_space)


def test_properties(mesh):
    P = FunctionSpace(mesh, "DG", 0)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    g = Function(V)

    def _check_extop_attributes_(x, ops, space, der, shape):
        assert x.ufl_function_space() == space
        assert x.ufl_operands == ops
        assert x.derivatives == der
        assert x.ufl_shape == shape

    f = lambda x, y: x*y
    pe = point_expr(f, function_space=P)
    pe2 = pe(u, g)

    _check_extop_attributes_(pe2, (u, g), P, (0, 0), ())

    assert pe2.operator_data == f
    assert pe2.expr == f


def test_pointwise_expr_operator(mesh):
    V = FunctionSpace(mesh, "CG", 1)

    x, y = SpatialCoordinate(mesh)

    v = Function(V).interpolate(sin(x))
    u = Function(V).interpolate(cos(x))

    p = point_expr(lambda x, y: x*y, function_space=V)
    p2 = p(u, v)

    assert p2.ufl_operands[0] == u
    assert p2.ufl_operands[1] == v
    assert p2.ufl_function_space() == V
    assert p2.derivatives == (0, 0)
    assert p2.ufl_shape == ()
    assert p2.expr(u, v) == u*v

    error = assemble((u*v-p2)**2*dx)
    assert error < 1.0e-3

    u2 = Function(V)
    g = Function(V).interpolate(cos(x))
    v = TestFunction(V)

    f = Function(V).interpolate(cos(x)*sin(y))
    p = point_expr(lambda x: x**2+1, function_space=V)
    p2 = p(g)

    F = (dot(grad(p2*u), grad(v)) + u*v)*dx - f*v*dx
    solve(F == 0, u)

    F2 = (dot(grad((g**2+1)*u2), grad(v)) + u2*v)*dx - f*v*dx
    solve(F2 == 0, u2)

    a1 = assemble(u*dx)
    a2 = assemble(u2*dx)
    err = (a1-a2)**2
    assert err < 1.0e-9


"""
def test_compute_derivatives(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    v = Function(V).interpolate(sin(x))
    u = Function(V).interpolate(cos(x))

    m = u*v
    a1 = m*dx

    p = point_expr(lambda x, y: 0.5*x**2*y, function_space=P)
    uhat = TrialFunction(P)
    p2 = p(u, v)
    dp2du = p2._ufl_expr_reconstruct_(u, v, derivatives=(1, 0), argument_slots=p2.argument_slots() + (uhat,))
    a2 = action(TrialFunction(P)*dx, dp2du*dx

    assert p2.ufl_operands[0] == u
    assert p2.ufl_operands[1] == v
    assert p2.ufl_function_space() == P
    assert dp2du.derivatives == (1, 0)
    assert p2.ufl_shape == ()
    assert p2.expr(u, v) == 0.5*u**2*v

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    # Not evaluate on the same space hence the lack of precision
    assert abs(assemble_a1 - assemble_a2) < 1.0e-3
"""


def test_scalar_check_equality(mesh):

    V1 = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    f = Function(V1).interpolate(cos(x)*sin(y))

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u2 = Function(V1)
    ps = point_expr(lambda x: x, function_space=V1)
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09

    # Action operator
    u2 = Function(V1)
    ps = action_point_expr(lambda x: x, function_space=V1)
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx

    # Check that an error is raised when we try to assemble the jacobian of the Global ExternalOperator ps
    check_error = False
    #try:
    #    solve(F2 == 0, u2)
    #except:
        # Should lead to a ValueError but as the error is raised in self.evaluate() in the assembly,
        # it leads to a ConvergenceError
    #    check_error = True
    #assert check_error
    print('\n\n\n Matfree !!')
    solve(F2 == 0, u2, solver_parameters={"mat_type": "matfree",
                                          "ksp_type": "cg",
                                          "pc_type": "none"})

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09


def test_vector_check_equality(mesh):

    V1 = VectorFunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    f = Function(V1).interpolate(as_vector([cos(x), sin(y)]))

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u2 = Function(V1)
    ps = point_expr(lambda x: x, function_space=V1)
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09

    # Action operator
    u2 = Function(V1)
    ps = action_point_expr(lambda x: x, function_space=V1)
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx

    # Check that an error is raised when we try to assemble the jacobian of the Global ExternalOperator ps
    #check_error = False
    #try:
    #    solve(F2 == 0, u2)
    #except:
        # Should lead to a ValueError but as the error is raised in self.evaluate() in the assembly,
        # it leads to a ConvergenceError
    #    check_error = True
    #assert check_error
    solve(F2 == 0, u2, solver_parameters={"mat_type": "matfree",
                                          "ksp_type": "cg",
                                          "pc_type": "none"})

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09


def test_tensor_check_equality(mesh):

    V0 = VectorFunctionSpace(mesh, "CG", 1)
    V1 = TensorFunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    phi = Function(V0).interpolate(as_vector([cos(x), sin(y)]))
    f = grad(phi)

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u2 = Function(V1)
    ps = point_expr(lambda x: x, function_space=V1)
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2, solver_parameters={"mat_type": "matfree",
                                          "ksp_type": "cg",
                                          "pc_type": "none"})

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09

    # Action operator
    u2 = Function(V1)
    ps = action_point_expr(lambda x: x, function_space=V1)
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx

    # Check that an error is raised when we try to assemble the jacobian of the Global ExternalOperator ps
    check_error = False
    try:
        solve(F2 == 0, u2)
    except:
        # Should lead to a ValueError but as the error is raised in self.evaluate() in the assembly,
        # it leads to a ConvergenceError
        check_error = True
    assert check_error
    solve(F2 == 0, u2, solver_parameters={"mat_type": "matfree",
                                          "ksp_type": "cg",
                                          "pc_type": "none"})

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09


def test_assemble_action(mesh):

    from ufl.algorithms.ad import expand_derivatives
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = Function(V).assign(1.)
    v = TestFunction(V)

    # External operators
    p = point_expr(lambda x: x, function_space=V)
    p = p(u)

    p_action = action_point_expr(lambda x: x, function_space=V)
    p_action = p_action(u)

    u_hat = TrialFunction(V)

    # Compute Jacobian
    a = inner(p, v)*dx
    Ja = derivative(a, u, u_hat)
    Ja = expand_derivatives(Ja)

    a_action = inner(p_action, v)*dx
    Ja_action = derivative(a_action, u, u_hat)
    Ja_action = expand_derivatives(Ja_action)

    # Assemble Ja and Ja_action
    assemble(Ja)

    # Check that an error is raised when we try to assemble the jacobian of the Global ExternalOperator
    check_error = False
    try:
        assemble(Ja_action)
    except ValueError:
        check_error = True
    assert check_error

    assemble(Ja_action, mat_type="matfree")

    # Check action arguments and arguments
    _test_action_arguments(Ja, Ja_action, u_hat, Function(V))


def _test_action_arguments(Ja, Ja_action, u_hat, g):

    dp, = Ja.external_operators()
    dp_action, = Ja_action.external_operators()

    assert dp.arguments() == ()
    assert dp.action_coefficients() == ()
    assert dp_action.arguments() == ((u_hat, False),)
    assert dp_action.action_coefficients() == ()

    # Take the action
    Ja = action(Ja, g)
    Ja_action = action(Ja_action, g)
    dp, = Ja.external_operators()
    dp_action, = Ja_action.external_operators()

    assert dp.arguments() == ()
    assert dp.action_coefficients() == ()
    assert dp_action.arguments() == ()
    assert dp_action.action_coefficients() == ((g, False),)
