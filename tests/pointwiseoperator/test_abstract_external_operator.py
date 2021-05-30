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

    def _check_extop_attributes_(x, ops, space, der, shape):
        assert x.ufl_function_space() == space
        assert x.ufl_operands == ops
        assert x.derivatives == der
        assert x.ufl_shape == shape

    f = lambda x, y: x*y
    p = TestAbstractExternalOperator(u, w, g, function_space=V, derivatives=(0, 0, 1), name='abstract_po', operator_data=f)

    _check_extop_attributes_(p, (u, w, g), V, (0, 0, 1), ())

    assert p.operator_data == f
    assert p._name == 'abstract_po'


def test_derivation_wrt_externaloperator(mesh):

    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    g = Function(V)
    v = TestFunction(V)
    u_hat = Function(V)

    p = TestAbstractExternalOperator(u, g, function_space=V)

    from ufl.algorithms.apply_derivatives import apply_derivatives

    l = sin(p**2)*v
    dl_dp = p*2.*cos(p**2)*v
    dl = diff(l, p)
    assert apply_derivatives(dl) == dl_dp

    L = p*u*dx
    dL_dp = u*u_hat*dx
    Gateaux_dL = derivative(L, p, u_hat)
    assert apply_derivatives(Gateaux_dL) == dL_dp


def test_add_dependencies(mesh):

    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    w = Function(V)

    e2 = TestAbstractExternalOperator(u, w, grad(u), div(w), function_space=V)
    der = [(0, 0, 0, 1), (1, 0, 0, 1), (2, 0, 1, 1)]
    args = [(), (), ()]
    e2.add_dependencies(der, args)
    de2 = tuple(e2.coefficient_dict.values())

    assert sorted(e2.coefficient_dict.keys()) == der
    assert der == sorted(e.derivatives for e in de2)
    assert de2[0]._extop_master == e2
    assert de2[1]._extop_master == e2
    assert de2[2]._extop_master == e2
