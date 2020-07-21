import numpy as np
import pytest
from firedrake import *

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
    space = request.param
    V_el = FiniteElement(space, mesh.ufl_cell(), degree, variant="integral")
    return FunctionSpace(mesh, V_el)


def test_div_curl_preserving(V):
    mesh = V.ufl_domain()
    dim = mesh.geometric_dimension()
    if dim == 2:
        x, y = SpatialCoordinate(mesh)
    elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)
    if dim == 2:
        expression = as_vector([cos(y), sin(x)])
    elif dim == 3:
        if "Nedelec" in V.ufl_element().family():
            expression = grad(sin(x)*exp(y)*z)
        else:
            expression = as_vector([sin(y)*z, cos(x)*z, exp(x)])

    f = interpolate(expression, V)
    if "Nedelec" in V.ufl_element().family():
        if dim == 2:
            # Skip this test
            norm_exp = 1e-15
        else:
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
        variant = "integral(" + str(degree+1) + ")"
        V_el = FiniteElement(space, mesh.ufl_cell(), degree, variant=variant)
        V = FunctionSpace(mesh, V_el)
        f = interpolate(expression, V)
        error_l2 = errornorm(expression, f, 'L2')
        if "Nedelec" in V.ufl_element().family():
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
def function_space(request, degree):
    return (request.param, degree)


def test_convergence_order(mesh, function_space):
    space, degree = function_space
    nref = 2
    nref_min = 1
    error = compute_interpolation_error(mesh, nref, space, degree)
    error_l2 = error.T[0]
    error_hD = error.T[1]
    conv_l2 = np.log(error_l2[0:-1]/error_l2[1:])/np.log(2)
    conv_hD = np.log(error_hD[0:-1]/error_hD[1:])/np.log(2)
    if "1" in space:
        conv_l2_expected = degree
    else:
        conv_l2_expected = degree + 1
    conv_hD_expected = degree
    assert np.allclose(conv_l2[nref_min:], conv_l2_expected*np.ones((nref-nref_min, 1)), atol=0.15)
    assert np.allclose(conv_hD[nref_min:], conv_hD_expected*np.ones((nref-nref_min, 1)), atol=0.15)
