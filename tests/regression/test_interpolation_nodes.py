import numpy as np
import pytest
from firedrake import *
from firedrake.__future__ import *

'''
The spaces N1div, N1curl, N2div and N2curl have the special property that the interpolation in these
spaces preserves the divergence/curl, cmp. [Mixed Finite Element Methods and Applications by
Daniele Boffi Franco Brezzi Michel Fortin, chapter 2] That means, e.g., if div(u) = 0 => div(I_{N1div} u)=0.
The test test_div_curl_preserving(V) tests this property.
The textbook convergence for integral degrees of freedom, cmp. [The FEniCs Book], is for
N1div, N1curl of degree q: L2 order q, Hdiv/curl order q
N2div, N2curl of degree q: L2 order q+1, Hdiv/curl order q
test_convergence_order(mesh, function_space) tests these rates.
'''


@pytest.fixture(params=["square", "cube"], scope="module")
def mesh(request):
    if request.param == "square":
        return SquareMesh(4, 4, 2)
    elif request.param == "cube":
        return CubeMesh(4, 4, 4, 2)


@pytest.fixture(params=[1, 2, 3], scope="module")
def degree(request):
    return request.param


@pytest.fixture(params=[("N1curl"),
                        ("N2curl"),
                        ("N1div"),
                        ("N2div")],
                ids=lambda x: "%s" % x)
def V(request, mesh, degree):
    family = request.param
    over_integration = max(0, 9 - degree)
    return FunctionSpace(mesh, family, degree, variant=f"integral({over_integration})")


def test_div_curl_preserving(V):
    mesh = V.mesh()
    dim = mesh.geometric_dimension()
    if dim == 2:
        x, y = SpatialCoordinate(mesh)
    elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)
    if dim == 2:
        if V.ufl_element().sobolev_space == HCurl:
            expression = grad(sin(x)*cos(y))
        else:
            expression = as_vector([cos(y), sin(x)])
    elif dim == 3:
        if V.ufl_element().sobolev_space == HCurl:
            expression = grad(sin(x)*exp(y)*z)
        else:
            expression = as_vector([sin(y)*z, cos(x)*z, exp(x)])

    f = assemble(interpolate(expression, V))
    if V.ufl_element().sobolev_space == HCurl:
        norm_exp = sqrt(assemble(inner(curl(f), curl(f))*dx))
    else:
        norm_exp = sqrt(assemble(inner(div(f), div(f))*dx))
    assert abs(norm_exp) < 1e-10


def compute_interpolation_error(baseMesh, nref, space, degree):
    mh = MeshHierarchy(baseMesh, nref)
    dim = mh[0].geometric_dimension()

    error = np.zeros((nref+1, 2))
    for i, mesh in enumerate(mh):
        if dim == 2:
            x, y = SpatialCoordinate(mesh)
        elif dim == 3:
            x, y, z = SpatialCoordinate(mesh)
        if dim == 2:
            expression = as_vector([sin(x)*cos(y), exp(x)*y])
        elif dim == 3:
            expression = as_vector([sin(y)*z*cos(x), cos(x)*z*x, exp(x)*y])
        V = FunctionSpace(mesh, space, degree, variant="integral")
        f = assemble(interpolate(expression, V))
        error_l2 = errornorm(expression, f, 'L2')
        if V.ufl_element().sobolev_space == HCurl:
            error_hD = errornorm(expression, f, 'hcurl')
        else:
            error_hD = errornorm(expression, f, 'hdiv')
        error[i] = np.array([error_l2, error_hD])
    return error


@pytest.fixture(params=[("N1curl"),
                        ("N2curl"),
                        ("N1div"),
                        ("N2div")],
                ids=lambda x: "%s" % x)
def space(request):
    return request.param


def expected_l2_order(space, degree):
    if space in {"N2curl", "N2div"}:
        return degree + 1
    else:
        return degree


def test_convergence_order(mesh, space, degree):
    nref = 2
    nref_min = 1
    error = compute_interpolation_error(mesh, nref, space, degree)
    error_l2 = error.T[0]
    error_hD = error.T[1]
    conv_l2 = np.log2(error_l2[0:-1]/error_l2[1:])
    conv_hD = np.log2(error_hD[0:-1]/error_hD[1:])
    conv_l2_expected = expected_l2_order(space, degree)
    conv_hD_expected = degree
    eps = 0.15
    assert all(conv_l2[nref_min:] > conv_l2_expected - eps)
    assert all(conv_hD[nref_min:] > conv_hD_expected - eps)
