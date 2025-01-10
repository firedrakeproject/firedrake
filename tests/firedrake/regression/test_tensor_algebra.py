import pytest
from firedrake import *
import numpy as np


@pytest.fixture(scope="module",
                params=["triangle",
                        "tet",
                        "quad"],
                ids=lambda x: "RT1(%s)" % x)
def mesh(request):
    if request.param == "triangle":
        return UnitTriangleMesh()
    elif request.param == "tet":
        return UnitTetrahedronMesh()
    elif request.param == "quad":
        return UnitSquareMesh(1, 1, quadrilateral=True)


@pytest.fixture(scope="module",
                params=[("mu_s*inner(grad(u), outer(conj(v), n)) * ds",
                         "mu_s*inner(grad(u), outer(conj(v), n)) * ds"),
                        ("mu_s*(-inner(grad(u), outer(conj(v), n), ) - inner(outer(conj(u), n), grad(v))) * ds",
                         "-2*mu_s*(inner(grad(u), outer(conj(v), n))) * ds"),
                        ("-mu_s*(inner(grad(u), outer(conj(v), n)) + inner(outer(conj(u), n), grad(v))) * ds",
                         "-2*mu_s*(inner(grad(u), outer(conj(v), n))) * ds"),
                        ("inner(dot(grad(u), mu), outer(conj(v), n)) * ds",
                         "mu_s*inner(grad(u), outer(conj(v), n)) * ds"),
                        ("(inner(dot(grad(u), mu), outer(conj(v), n)) + inner(outer(conj(u), n), dot(grad(v), mu))) * ds",
                         "2*mu_s*inner(grad(u), outer(conj(v), n)) * ds"),
                        ("-(inner(dot(grad(u), mu), outer(conj(v), n)) + inner(outer(conj(u), n), dot(grad(v), mu))) * ds",
                         "-2*mu_s*inner(grad(u), outer(conj(v), n)) * ds"),
                        ("(-inner(dot(grad(u), mu), outer(conj(v), n)) - inner(outer(conj(u), n), dot(grad(v), mu))) * ds",
                         "-2*mu_s*inner(grad(u), outer(conj(v), n)) * ds")],
                ids=lambda x: x[0])
def form_expect(request, mesh):
    dim = mesh.geometric_dimension()
    if mesh.ufl_cell().cellname() == "quadrilateral":
        V = FunctionSpace(mesh, "RTCF", 1)
    else:
        V = FunctionSpace(mesh, "RT", 1)

    mu_s = Constant(1.0)
    mu = as_tensor(np.diag(np.repeat(mu_s, dim)))  # noqa

    n = FacetNormal(mesh)       # noqa
    u = TrialFunction(V)        # noqa
    v = TestFunction(V)         # noqa

    form, expect = request.param
    return eval(form), eval(expect)


def test_tensor_algebra_simplification(form_expect):
    form, expect = form_expect

    expect = assemble(expect).M.values

    actual = assemble(form).M.values

    assert np.allclose(expect, actual)
