import pytest
from firedrake import *
from firedrake.slate.slac import compile_expression as compile_slate


@pytest.fixture(scope='module', params=[interval,
                                        triangle,
                                        tetrahedron,
                                        quadrilateral])
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


def test_determinism_and_caching(tensor):
    """Tests that the :meth:'compile_slate_expression' forms
    a numerically deterministic system. That is, produced kernels
    are consistent.

    This test also checks that the caching mechanism is functioning
    properly.
    """
    A = tensor
    # Reconstruct an identical tensor, but as a different instance
    # of a Tensor
    B = Tensor(tensor.form)
    kernel1 = compile_slate(A)
    kernel2 = compile_slate(B)

    # Checking equivalence of kernels
    assert kernel1 is kernel2

    # Changing TSFC parameters should change kernel
    kernel3 = compile_slate(B, {"mode": "vanilla"})
    assert kernel3 is not kernel2

    for k1, k2 in zip(kernel1, kernel3):
        assert k1 is not k2
