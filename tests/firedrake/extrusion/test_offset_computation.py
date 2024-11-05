import pytest
from firedrake import *


def test_no_offset_zero():
    m = UnitSquareMesh(1, 1)
    m = ExtrudedMesh(m, layers=2)

    V = FunctionSpace(m, 'CG', 2)

    assert (V.exterior_facet_node_map().offset != 0).all()


def test_offset_p2():
    m = UnitSquareMesh(1, 1)
    m = ExtrudedMesh(m, layers=1)

    V = FunctionSpace(m, 'CG', 2)

    assert (V.exterior_facet_node_map().offset == 2).all()


def test_offset_enriched():
    m = UnitSquareMesh(1, 1)
    m = ExtrudedMesh(m, layers=1)

    ele = TensorProductElement(FiniteElement("CG", "triangle", 2),
                               FiniteElement("CG", "interval", 1)) + \
        TensorProductElement(FiniteElement("CG", "triangle", 1),
                             FiniteElement("DG", "interval", 0))

    V = FunctionSpace(m, ele)

    # On each facet we have:
    #
    #  o--x--o
    #  |     |
    #  o     o
    #  |     |
    #  o--x--o
    assert (V.exterior_facet_node_map().offset
            == [2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2]).all()


def run_offset_parallel():
    m = UnitSquareMesh(20, 20)
    m = ExtrudedMesh(m, layers=1)

    V = FunctionSpace(m, 'CG', 2)

    offset = V.exterior_facet_node_map().offset

    offsets = m.comm.allgather(offset)
    assert all((o == offset).all() for o in offsets)


@pytest.mark.parallel(nprocs=2)
def test_offset_parallel_allsame():
    run_offset_parallel()
