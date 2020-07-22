import pytest
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


def test_abstract_pointwise_operator(mesh):

    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    w = Function(V)
    g = Function(V)

    def _check_extop_attributes_(x, ops, space, der, shape):
        assert x.ufl_function_space() == space
        assert x.ufl_operands == ops
        assert x.derivatives == der
        assert x.ufl_shape == shape

    class TestAbstract(AbstractExternalOperator):

        _external_operator_type = 'LOCAL'

        def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data):
            AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, operator_data=operator_data)

    f = lambda x, y: x*y
    p = TestAbstract(u, w, g, function_space=V, derivatives=(0, 0, 1), count=9999,
                     name='abstract_po', operator_data=f)

    _check_extop_attributes_(p, (u, w, g), V, (0, 0, 1), ())

    assert p.operator_data == f
    assert p._count == 9999
    assert p._name == 'abstract_po'


def test_derivation_wrt_pointwiseoperator(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    u = Function(V)
    g = Function(V)
    v = TestFunction(V)
    u_hat = Function(V)

    p = point_expr(lambda x, y: x*y, function_space=P)
    p2 = p(u, g)

    from ufl.algorithms.apply_derivatives import apply_derivatives

    l = sin(p2**2)*v
    dl_dp2 = p2*2.*cos(p2**2)*v
    dl = diff(l, p2)
    assert apply_derivatives(dl) == dl_dp2

    L = p2*u*dx
    dL_dp2 = u*u_hat*dx
    Gateaux_dL = derivative(L, p2, u_hat)
    assert apply_derivatives(Gateaux_dL) == dL_dp2
