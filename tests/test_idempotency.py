import ufl
from tsfc import compile_form
import pytest


@pytest.fixture(params=[ufl.interval,
                        pytest.mark.xfail(reason="indices")(ufl.triangle),
                        pytest.mark.xfail(reason="indices")(ufl.quadrilateral),
                        pytest.mark.xfail(reason="indices")(ufl.tetrahedron)],
                ids=lambda x: x.cellname())
def cell(request):
    return request.param


@pytest.fixture
def mesh(cell):
    c = ufl.VectorElement("CG", cell, 2)
    return ufl.Mesh(c)


@pytest.fixture(params=[ufl.FiniteElement,
                        ufl.VectorElement,
                        ufl.TensorElement],
                ids=["FE", "VE", "TE"])
def V(request, mesh):
    return ufl.FunctionSpace(mesh, request.param("CG", mesh.ufl_cell(), 2))


@pytest.fixture(params=["functional", "1-form", "2-form"])
def form(V, request):
    if request.param == "functional":
        u = ufl.Coefficient(V)
        v = ufl.Coefficient(V)
    elif request.param == "1-form":
        u = ufl.Coefficient(V)
        v = ufl.TestFunction(V)
    elif request.param == "2-form":
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

    return ufl.inner(u, v)*ufl.dx


def test_idempotency(form):
    k1 = compile_form(form)[0]
    k2 = compile_form(form)[0]

    assert k1.ast.gencode() == k2.ast.gencode()


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
