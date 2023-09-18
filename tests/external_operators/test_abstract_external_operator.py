import pytest
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


class TestAbstractExternalOperator(AbstractExternalOperator):
    def __init__(self, *operands, function_space, derivatives=None, **kwargs):
        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, **kwargs)


def test_abstract_external_operator(mesh):

    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    w = Function(V)
    g = Function(V)
    ghat = TrialFunction(V)

    def _check_extop_attributes_(x, ops, space, der, shape):
        assert x.ufl_function_space() == space
        assert x.ufl_operands == ops
        assert x.derivatives == der
        assert x.ufl_shape == shape

    f = lambda x, y: x*y
    args = (Argument(V, 0), ghat)
    p = TestAbstractExternalOperator(u, w, g, function_space=V, derivatives=(0, 0, 1), argument_slots=args,
                                     operator_data={"f": f})

    _check_extop_attributes_(p, (u, w, g), V, (0, 0, 1), ())

    assert p.operator_data == {"f": f}


def test_derivation_wrt_externaloperator(mesh):

    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    g = Function(V)
    u_hat = Function(V)

    p = TestAbstractExternalOperator(u, g, function_space=V)

    from ufl.algorithms.apply_derivatives import apply_derivatives

    L = p*u*dx
    dL_dp = u*u_hat*dx
    Gateaux_dL = derivative(L, p, u_hat)
    assert apply_derivatives(Gateaux_dL) == dL_dp


def test_assemble_methods(mesh):

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    g = Function(V)

    class MyExternalOperator1(AbstractExternalOperator):
        def __init__(self, *args, **kwargs):
            AbstractExternalOperator.__init__(self, *args, **kwargs)

        @assemble_method((1, 0), (0, 1))
        def dN_du(self, *args, **kwargs):
            """Implements dN/du(u, m; uhat, v*), which is represented by derivs = (1, 0) and args = (0, 1)"""
            pass

    e1 = MyExternalOperator1(u, g, function_space=V)
    assert e1._assembly_registry == MyExternalOperator1._assembly_registry
    assert MyExternalOperator1._assembly_registry == {((1, 0), (0, 1)): MyExternalOperator1.dN_du}

    class MyExternalOperator2(MyExternalOperator1):
        def __init__(self, *args, **kwargs):
            MyExternalOperator1.__init__(self, *args, **kwargs)

        @assemble_method((0, 1), (1, 0))
        def dN_dm_adjoint(self, *args, **kwargs):
            """Implements dN/dm^{*}(u, m; v*, mhat), which is represented by derivs = (0, 1) and args = (1, 0)"""
            pass

        @assemble_method(1, (0, 1))
        def jacobian(self, *args, **kwargs):
            """Implements the external operator jacobian: i.e. dN/dop with op in N's operands"""
            pass

    e2 = MyExternalOperator2(g, u*g, function_space=V)
    assert e2._assembly_registry == MyExternalOperator2._assembly_registry
    assert MyExternalOperator2._assembly_registry == {((1, 0), (0, 1)): MyExternalOperator1.dN_du,
                                                      ((0, 1), (1, 0)): MyExternalOperator2.dN_dm_adjoint,
                                                      (1, (0, 1)): MyExternalOperator2.jacobian}
