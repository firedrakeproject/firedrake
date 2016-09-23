from __future__ import absolute_import, print_function, division
from tsfc import fiatinterface as f
import pytest
import ufl


@pytest.fixture(params=["BDM",
                        "BDFM",
                        "DRT",
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
    element = f.create_element(ufl_element)
    assert isinstance(element, f.supported_elements[ufl_element.family()])


@pytest.fixture
def ufl_vector_element(triangle_names):
    return ufl.VectorElement(triangle_names, ufl.triangle, 2)


@pytest.mark.parametrize("mixed",
                         [False, True])
def test_triangle_vector(mixed, ufl_element, ufl_vector_element):
    scalar = f.create_element(ufl_element)
    vector = f.create_element(ufl_vector_element, vector_is_mixed=mixed)

    if not mixed:
        assert isinstance(scalar, f.supported_elements[ufl_element.family()])
        assert isinstance(vector, f.supported_elements[ufl_element.family()])

    else:
        assert isinstance(vector, f.MixedElement)
        assert isinstance(vector.elements()[0], f.supported_elements[ufl_element.family()])
        assert len(vector.elements()) == ufl_vector_element.num_sub_elements()


@pytest.fixture(params=["CG", "DG"])
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

    tensor = f.create_element(tensor_ufl)
    A = f.create_element(ufl_A)
    B = f.create_element(ufl_B)

    assert isinstance(tensor, f.supported_elements[tensor_ufl.family()])

    assert tensor.A is A
    assert tensor.B is B


def test_cache_hit(ufl_element):
    A = f.create_element(ufl_element)
    B = f.create_element(ufl_element)

    assert A is B


def test_cache_hit_vector(ufl_vector_element):
    A = f.create_element(ufl_vector_element)
    B = f.create_element(ufl_vector_element)

    assert A is B

    assert all(a == A.elements()[0] for a in A.elements())


def test_cache_miss_vector(ufl_vector_element):
    A = f.create_element(ufl_vector_element)
    B = f.create_element(ufl_vector_element, vector_is_mixed=False)

    assert A is not B

    assert A.elements()[0] is not B


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
