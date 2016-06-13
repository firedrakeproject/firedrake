from tsfc import quadrature as q
import ufl
import pytest


def test_invalid_quadrature_rule():
    with pytest.raises(ValueError):
        q.QuadratureRule([[0.5, 0.5]], [0.5, 0.5, 0.5])


@pytest.fixture(params=["interval", "triangle",
                        "tetrahedron", "quadrilateral"])
def cell(request):
    cells = {"interval": ufl.interval,
             "triangle": ufl.triangle,
             "tetrahedron": ufl.tetrahedron,
             "quadrilateral": ufl.quadrilateral}

    return cells[request.param]


@pytest.fixture
def tensor_product_cell(cell):
    if cell.cellname() == "tetrahedron":
        pytest.skip("Tensor-producted tet not supported")

    return ufl.TensorProductCell(cell, ufl.interval)


@pytest.mark.parametrize("degree",
                         [1, 2])
@pytest.mark.parametrize("itype",
                         ["cell", "interior_facet", "exterior_facet"])
def test_select_degree(cell, degree, itype):
    selected = q.select_degree(degree, cell, itype)
    assert selected == degree


@pytest.mark.parametrize("degree",
                         [(1, 2), (2, 3)])
@pytest.mark.parametrize("itype",
                         ["interior_facet_horiz", "exterior_facet_top",
                          "exterior_facet_bottom"])
def test_select_degree_horiz_facet(tensor_product_cell, degree, itype):
    selected = q.select_degree(degree, tensor_product_cell, itype)
    assert selected == degree[0]


@pytest.mark.parametrize("degree",
                         [(1, 2), (2, 3)])
@pytest.mark.parametrize("itype",
                         ["interior_facet_vert", "exterior_facet_vert"])
def test_select_degree_vert_facet(tensor_product_cell, degree, itype):
    selected = q.select_degree(degree, tensor_product_cell, itype)
    if tensor_product_cell.topological_dimension() == 2:
        assert selected == degree[1]
    else:
        assert selected == degree


@pytest.mark.parametrize("itype",
                         ["interior_facet_horiz",
                          "interior_facet_vert",
                          "exterior_facet_vert",
                          "exterior_facet_top",
                          "exterior_facet_bottom",
                          "nonsense"])
def test_invalid_integral_type(cell, itype):
    with pytest.raises(ValueError):
        q.select_degree(1, cell, itype)


@pytest.mark.parametrize("itype",
                         ["interior_facet",
                          "exterior_facet",
                          "nonsense"])
def test_invalid_integral_type_tensor_prod(tensor_product_cell, itype):
    with pytest.raises(ValueError):
        q.select_degree((1, 1), tensor_product_cell, itype)


@pytest.mark.parametrize("itype",
                         ["interior_facet",
                          "exterior_facet",
                          "cell"])
@pytest.mark.parametrize("scheme",
                         ["default",
                          "canonical"])
def test_invalid_quadrature_degree(cell, itype, scheme):
    with pytest.raises(ValueError):
        q.create_quadrature(cell, itype, -1, scheme)


@pytest.mark.parametrize("itype",
                         ["interior_facet_horiz",
                          "interior_facet_vert",
                          "exterior_facet_vert",
                          "exterior_facet_top",
                          "exterior_facet_bottom",
                          "cell"])
def test_invalid_quadrature_degree_tensor_prod(tensor_product_cell, itype):
    with pytest.raises(ValueError):
        q.create_quadrature(tensor_product_cell, itype, (-1, -1))


def test_high_degree_runtime_error(cell):
    with pytest.raises(RuntimeError):
        q.create_quadrature(cell, "cell", 60)


def test_high_degree_runtime_error_tensor_prod(tensor_product_cell):
    with pytest.raises(RuntimeError):
        q.create_quadrature(tensor_product_cell, "cell", (60, 60))


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
