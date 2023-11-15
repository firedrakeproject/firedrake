import ufl
import finat.ufl
from tsfc import compile_form
import loopy
import pytest


@pytest.fixture(params=[ufl.interval,
                        ufl.triangle,
                        ufl.quadrilateral,
                        ufl.tetrahedron],
                ids=lambda x: x.cellname())
def cell(request):
    return request.param


@pytest.fixture(params=[1, 2],
                ids=lambda x: "P%d-coords" % x)
def coord_degree(request):
    return request.param


@pytest.fixture
def mesh(cell, coord_degree):
    c = finat.ufl.VectorElement("CG", cell, coord_degree)
    return ufl.Mesh(c)


@pytest.fixture(params=[finat.ufl.FiniteElement,
                        finat.ufl.VectorElement,
                        finat.ufl.TensorElement],
                ids=["FE", "VE", "TE"])
def V(request, mesh):
    return ufl.FunctionSpace(mesh, request.param("CG", mesh.ufl_cell(), 2))


@pytest.fixture(params=["cell", "ext_facet", "int_facet"])
def itype(request):
    return request.param


@pytest.fixture(params=["functional", "1-form", "2-form"])
def form(V, itype, request):
    if request.param == "functional":
        u = ufl.Coefficient(V)
        v = ufl.Coefficient(V)
    elif request.param == "1-form":
        u = ufl.Coefficient(V)
        v = ufl.TestFunction(V)
    elif request.param == "2-form":
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

    if itype == "cell":
        return ufl.inner(u, v)*ufl.dx
    elif itype == "ext_facet":
        return ufl.inner(u, v)*ufl.ds
    elif itype == "int_facet":
        return ufl.inner(u('+'), v('-'))*ufl.dS


def test_idempotency(form):
    k1 = compile_form(form)[0]
    k2 = compile_form(form)[0]

    assert loopy.generate_code_v2(k1.ast).device_code() == loopy.generate_code_v2(k2.ast).device_code()


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
