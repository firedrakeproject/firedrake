from firedrake import *
import pytest


@pytest.fixture(scope='module')
def quad_mesh():
    return ExtrudedMesh(UnitIntervalMesh(1), 1)


@pytest.fixture(scope='module')
def prism_mesh():
    return ExtrudedMesh(UnitTriangleMesh(), 1)


@pytest.fixture(scope='module')
def hex_mesh():
    return ExtrudedMesh(UnitSquareMesh(1, 1, quadrilateral=True), 1)


@pytest.mark.parametrize(('args', 'kwargs', 'horiz_expected', 'vert_expected'),
                         [(("DQ", 0), dict(),
                           {0: [0], 1: [0]},
                           {0: [0], 1: [0]}),
                          (("DQ", 1), dict(),
                           {0: [0, 2], 1: [1, 3]},
                           {0: [0, 1], 1: [2, 3]}),
                          (("Q", 1), dict(),
                           {0: [0, 2], 1: [1, 3]},
                           {0: [0, 1], 1: [2, 3]}),
                          (("DG", 0), dict(vfamily="CG", vdegree=1),
                           {0: [0], 1: [1]},
                           {0: [0, 1], 1: [0, 1]}),
                          (("CG", 1), dict(vfamily="DG", vdegree=0),
                           {0: [0, 1], 1: [0, 1]},
                           {0: [0], 1: [1]}),
                          (("RTCF", 1), dict(),
                           {0: [0, 1, 2], 1: [0, 1, 3]},
                           {0: [0, 2, 3], 1: [1, 2, 3]}),
                          (("RTCE", 1), dict(),
                           {0: [0, 1, 2], 1: [0, 1, 3]},
                           {0: [0, 2, 3], 1: [1, 2, 3]})])
def test_quad(quad_mesh, args, kwargs, horiz_expected, vert_expected):
    V = FunctionSpace(quad_mesh, *args, **kwargs)
    assert horiz_expected == V.fiat_element.horiz_facet_support_dofs()
    assert vert_expected == V.fiat_element.vert_facet_support_dofs()


@pytest.mark.parametrize(('args', 'kwargs', 'horiz_expected', 'vert_expected'),
                         [(("DG", 0), dict(),
                           {0: [0], 1: [0]},
                           {0: [0], 1: [0], 2: [0]}),
                          (("DG", 1), dict(),
                           {0: [0, 2, 4], 1: [1, 3, 5]},
                           {0: [2, 3, 4, 5], 1: [0, 1, 4, 5], 2: [0, 1, 2, 3]}),
                          (("CG", 1), dict(),
                           {0: [0, 2, 4], 1: [1, 3, 5]},
                           {0: [2, 3, 4, 5], 1: [0, 1, 4, 5], 2: [0, 1, 2, 3]}),
                          (("DG", 0), dict(vfamily="CG", vdegree=1),
                           {0: [0], 1: [1]},
                           {0: [0, 1], 1: [0, 1], 2: [0, 1]}),
                          (("CG", 1), dict(vfamily="DG", vdegree=0),
                           {0: [0, 1, 2], 1: [0, 1, 2]},
                           {0: [1, 2], 1: [0, 2], 2: [0, 1]})])
def test_prism(prism_mesh, args, kwargs, horiz_expected, vert_expected):
    V = FunctionSpace(prism_mesh, *args, **kwargs)
    assert horiz_expected == V.fiat_element.horiz_facet_support_dofs()
    assert vert_expected == V.fiat_element.vert_facet_support_dofs()


@pytest.mark.parametrize(('space', 'degree', 'horiz_expected', 'vert_expected'),
                         [("RT", 1,
                           {0: [0, 1, 2, 3], 1: [0, 1, 2, 4]},
                           {0: range(5), 1: range(5), 2: range(5)}),
                          ("BDM", 1,
                           {0: [0, 1, 2, 3, 4, 5, 6], 1: [0, 1, 2, 3, 4, 5, 7]},
                           {0: range(8), 1: range(8), 2: range(8)})])
def test_prism_hdiv(prism_mesh, space, degree, horiz_expected, vert_expected):
    W0_h = FiniteElement(space, "triangle", degree)
    W1_h = FiniteElement("DG", "triangle", degree - 1)

    W0_v = FiniteElement("DG", "interval", degree - 1)
    W0 = HDiv(OuterProductElement(W0_h, W0_v))

    W1_v = FiniteElement("CG", "interval", degree)
    W1 = HDiv(OuterProductElement(W1_h, W1_v))

    V = FunctionSpace(prism_mesh, W0+W1)
    assert horiz_expected == V.fiat_element.horiz_facet_support_dofs()
    assert vert_expected == V.fiat_element.vert_facet_support_dofs()


@pytest.mark.parametrize(('space', 'degree', 'horiz_expected', 'vert_expected'),
                         [("RT", 1,
                           {0: [0, 1, 2, 3, 5, 7], 1: [0, 1, 2, 4, 6, 8]},
                           {0: [1, 2] + range(3, 9),
                            1: [0, 2] + range(3, 9),
                            2: [0, 1] + range(3, 9)}),
                          ("BDM", 1,
                           {0: range(3) + range(3, 15, 2), 1: range(3) + range(4, 15, 2)},
                           {0: [1, 2] + range(3, 15),
                            1: [0, 2] + range(3, 15),
                            2: [0, 1] + range(3, 15)})])
def test_prism_hcurl(prism_mesh, space, degree, horiz_expected, vert_expected):
    W0_h = FiniteElement("CG", "triangle", degree)
    W1_h = FiniteElement(space, "triangle", degree)

    W0_v = FiniteElement("DG", "interval", degree - 1)
    W0 = HCurl(OuterProductElement(W0_h, W0_v))

    W1_v = FiniteElement("CG", "interval", degree)
    W1 = HCurl(OuterProductElement(W1_h, W1_v))

    V = FunctionSpace(prism_mesh, W0+W1)
    assert horiz_expected == V.fiat_element.horiz_facet_support_dofs()
    assert vert_expected == V.fiat_element.vert_facet_support_dofs()


@pytest.mark.parametrize(('args', 'kwargs', 'horiz_expected', 'vert_expected'),
                         [(("DQ", 0), dict(),
                           {0: [0], 1: [0]},
                           {0: [0], 1: [0], 2: [0], 3: [0]}),
                          (("DQ", 1), dict(),
                           {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]},
                           {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [0, 1, 4, 5], 3: [2, 3, 6, 7]}),
                          (("Q", 1), dict(),
                           {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]},
                           {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [0, 1, 4, 5], 3: [2, 3, 6, 7]}),
                          (("DQ", 0), dict(vfamily="CG", vdegree=1),
                           {0: [0], 1: [1]},
                           {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1]}),
                          (("Q", 1), dict(vfamily="DG", vdegree=0),
                           {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]},
                           {0: [0, 1], 1: [2, 3], 2: [0, 2], 3: [1, 3]}),
                          (("NCF", 1), dict(),
                           {0: [0, 1, 2, 3, 4], 1: [0, 1, 2, 3, 5]},
                           {0: [0, 2, 3, 4, 5],
                            1: [1, 2, 3, 4, 5],
                            2: [0, 1, 2, 4, 5],
                            3: [0, 1, 3, 4, 5]}),
                          (("NCE", 1), dict(),
                           {0: [0, 1, 2, 3, 4, 6, 8, 10], 1: [0, 1, 2, 3, 5, 7, 9, 11]},
                           {0: [0, 1, 4, 5, 8, 9, 10, 11],
                            1: [2, 3, 6, 7, 8, 9, 10, 11],
                            2: [0, 2, 4, 5, 6, 7, 8, 9],
                            3: [1, 3, 4, 5, 6, 7, 10, 11]})])
def test_hex(hex_mesh, args, kwargs, horiz_expected, vert_expected):
    V = FunctionSpace(hex_mesh, *args, **kwargs)
    assert horiz_expected == V.fiat_element.horiz_facet_support_dofs()
    assert vert_expected == V.fiat_element.vert_facet_support_dofs()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
