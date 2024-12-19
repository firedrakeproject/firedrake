import gem
import numpy
from finat.point_set import PointSet
from gem.interpreter import evaluate
from finat.element_factory import create_element
from ufl import quadrilateral
from finat.ufl import FiniteElement, RestrictedElement


def test_issue_274():
    # See https://github.com/firedrakeproject/tsfc/issues/274
    ufl_element = RestrictedElement(
        FiniteElement("Q", quadrilateral, 2), restriction_domain="facet"
    )
    ps = PointSet([[0.5]])
    finat_element = create_element(ufl_element)
    evaluations = []
    for eid in range(4):
        (val,) = finat_element.basis_evaluation(0, ps, (1, eid)).values()
        evaluations.append(val)

    i = gem.Index()
    j = gem.Index()
    (expr,) = evaluate(
        [
            gem.ComponentTensor(
                gem.Indexed(gem.select_expression(evaluations, i), (j,)),
                (*ps.indices, i, j),
            )
        ]
    )

    (expect,) = evaluate(
        [
            gem.ComponentTensor(
                gem.Indexed(gem.ListTensor(evaluations), (i, j)), (*ps.indices, i, j)
            )
        ]
    )

    assert numpy.allclose(expr.arr, expect.arr)
