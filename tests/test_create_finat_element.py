import pytest

import ufl
import finat
from tsfc.finatinterface import create_element, supported_elements


@pytest.fixture(params=["BDM",
                        "BDFM",
                        "Lagrange",
                        "N1curl",
                        "N2curl",
                        "RT",
                        "Regge"])
def triangle_names(request):
    return request.param


@pytest.fixture
def ufl_element(triangle_names):
    return ufl.FiniteElement(triangle_names, ufl.triangle, 2)


def test_triangle_basic(ufl_element):
    element = create_element(ufl_element)
    assert isinstance(element, supported_elements[ufl_element.family()])


@pytest.fixture
def ufl_vector_element(triangle_names):
    return ufl.VectorElement(triangle_names, ufl.triangle, 2)


def test_triangle_vector(ufl_element, ufl_vector_element):
    scalar = create_element(ufl_element)
    vector = create_element(ufl_vector_element)

    assert isinstance(vector, finat.TensorFiniteElement)
    assert scalar == vector.base_element


@pytest.fixture(params=["CG", "DG", "DG L2"])
def tensor_name(request):
    return request.param


@pytest.fixture(params=[ufl.interval, ufl.triangle,
                        ufl.quadrilateral],
                ids=lambda x: x.cellname())
def ufl_A(request, tensor_name):
    return ufl.FiniteElement(tensor_name, request.param, 1)


@pytest.fixture
def ufl_B(tensor_name):
    return ufl.FiniteElement(tensor_name, ufl.interval, 1)


def test_tensor_prod_simple(ufl_A, ufl_B):
    tensor_ufl = ufl.TensorProductElement(ufl_A, ufl_B)

    tensor = create_element(tensor_ufl)
    A = create_element(ufl_A)
    B = create_element(ufl_B)

    assert isinstance(tensor, finat.TensorProductElement)

    assert tensor.factors == (A, B)


@pytest.mark.parametrize(('family', 'expected_cls'),
                         [('P', finat.GaussLobattoLegendre),
                          ('DP', finat.GaussLegendre),
                          ('DP L2', finat.GaussLegendre)])
def test_interval_variant_default(family, expected_cls):
    ufl_element = ufl.FiniteElement(family, ufl.interval, 3)
    assert isinstance(create_element(ufl_element), expected_cls)


@pytest.mark.parametrize(('family', 'variant', 'expected_cls'),
                         [('P', 'equispaced', finat.Lagrange),
                          ('P', 'spectral', finat.GaussLobattoLegendre),
                          ('DP', 'equispaced', finat.DiscontinuousLagrange),
                          ('DP', 'spectral', finat.GaussLegendre),
                          ('DP L2', 'equispaced', finat.DiscontinuousLagrange),
                          ('DP L2', 'spectral', finat.GaussLegendre)])
def test_interval_variant(family, variant, expected_cls):
    ufl_element = ufl.FiniteElement(family, ufl.interval, 3, variant=variant)
    assert isinstance(create_element(ufl_element), expected_cls)


def test_triangle_variant_spectral():
    ufl_element = ufl.FiniteElement('DP', ufl.triangle, 2, variant='spectral')
    create_element(ufl_element)


def test_triangle_variant_spectral_l2():
    ufl_element = ufl.FiniteElement('DP L2', ufl.triangle, 2, variant='spectral')
    create_element(ufl_element)


def test_quadrilateral_variant_spectral_q():
    element = create_element(ufl.FiniteElement('Q', ufl.quadrilateral, 3, variant='spectral'))
    assert isinstance(element.product.factors[0], finat.GaussLobattoLegendre)
    assert isinstance(element.product.factors[1], finat.GaussLobattoLegendre)


def test_quadrilateral_variant_spectral_dq():
    element = create_element(ufl.FiniteElement('DQ', ufl.quadrilateral, 1, variant='spectral'))
    assert isinstance(element.product.factors[0], finat.GaussLegendre)
    assert isinstance(element.product.factors[1], finat.GaussLegendre)


def test_quadrilateral_variant_spectral_dq_l2():
    element = create_element(ufl.FiniteElement('DQ L2', ufl.quadrilateral, 1, variant='spectral'))
    assert isinstance(element.product.factors[0], finat.GaussLegendre)
    assert isinstance(element.product.factors[1], finat.GaussLegendre)


def test_cache_hit(ufl_element):
    A = create_element(ufl_element)
    B = create_element(ufl_element)

    assert A is B


def test_cache_hit_vector(ufl_vector_element):
    A = create_element(ufl_vector_element)
    B = create_element(ufl_vector_element)

    assert A is B


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
