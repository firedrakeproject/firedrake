import pytest
from firedrake import *
from firedrake.slate import compile_expression as compile_slate


@pytest.fixture(scope='module', params=[interval, triangle, tetrahedron, quadrilateral])
def mesh(request):
    cell = request.param
    if cell == interval:
        return UnitIntervalMesh(1)
    elif cell == triangle:
        return UnitSquareMesh(1, 1)
    elif cell == tetrahedron:
        return UnitCubeMesh(1, 1, 1)
    elif cell == quadrilateral:
        return UnitSquareMesh(1, 1, quadrilateral=True)
    else:
        raise ValueError("%s cell not recognized" % cell)


@pytest.fixture(scope='module', params=['dg0', 'dg1'])
def V(request, mesh):
    dg0 = FunctionSpace(mesh, "DG", 0)
    dg1 = FunctionSpace(mesh, "DG", 1)
    return {'dg0': dg0,
            'dg1': dg1}[request.param]


@pytest.fixture(scope='module', params=["cell",
                                        "exterior_facet",
                                        "interior_facet"])
def int_type(request):
    return request.param


@pytest.fixture(scope='module', params=["rank_one", "rank_two"])
def tensor(V, int_type, request):
    if request.param == "rank_one":
        u = Coefficient(V)
        v = TestFunction(V)
    elif request.param == "rank_two":
        u = TrialFunction(V)
        v = TestFunction(V)
    else:
        raise ValueError("Not recognized parameter: %s" % request.param)

    measure = {"cell": dx,
               "interior_facet": dS,
               "exterior_facet": ds}

    return Tensor(inner(u, v) * measure[int_type])


def test_determinism(tensor):
    """Tests that the :meth:'compile_slate_expression' forms
    a numerically deterministic system. That is, produced kernels
    are consistent.
    """
    kernel1 = compile_slate(tensor)
    kernel2 = compile_slate(tensor)

    # Checking equivalence of kernels
    assert kernel1[0].kinfo.kernel._ast == kernel2[0].kinfo.kernel._ast


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
