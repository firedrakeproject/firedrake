from __future__ import absolute_import, print_function, division
from tsfc import finatinterface as f
import pytest
import ufl
import finat


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


def test_triangle_vector(ufl_element, ufl_vector_element):
    scalar = f.create_element(ufl_element)
    vector = f.create_element(ufl_vector_element)

    assert isinstance(vector, finat.TensorFiniteElement)
    assert scalar == vector.base_element


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

    assert isinstance(tensor, finat.TensorProductElement)

    assert tensor.factors == (A, B)


def test_cache_hit(ufl_element):
    A = f.create_element(ufl_element)
    B = f.create_element(ufl_element)

    assert A is B


def test_cache_hit_vector(ufl_vector_element):
    A = f.create_element(ufl_vector_element)
    B = f.create_element(ufl_vector_element)

    assert A is B


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
