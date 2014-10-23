import pytest
from firedrake import *


@pytest.mark.xfail
def test_no_offset_zero():
    m = UnitSquareMesh(1, 1)
    m = ExtrudedMesh(m, layers=2)

    V = FunctionSpace(m, 'CG', 2)

    assert (V.exterior_facet_boundary_node_map("topological").offset != 0).all()


@pytest.mark.xfail
def test_offset_p2():
    m = UnitSquareMesh(1, 1)
    m = ExtrudedMesh(m, layers=1)

    V = FunctionSpace(m, 'CG', 2)

    assert (V.exterior_facet_boundary_node_map("topological").offset == 2).all()


@pytest.mark.xfail
def test_offset_enriched():
    m = UnitSquareMesh(1, 1)
    m = ExtrudedMesh(m, layers=1)

    ele = OuterProductElement(FiniteElement("CG", "triangle", 2),
                              FiniteElement("CG", "interval", 1)) + \
        OuterProductElement(FiniteElement("CG", "triangle", 1),
                            FiniteElement("DG", "interval", 0))

    V = FunctionSpace(m, ele)

    # On each facet we have:
    #
    #  o--x--o
    #  |     |
    #  o     o
    #  |     |
    #  o--x--o
    #
    # Where the numbering is such that the two "x" dofs are numbered last.
    assert (V.exterior_facet_boundary_node_map("topological").offset ==
            [2, 2, 2, 2, 2, 2, 1, 1]).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
