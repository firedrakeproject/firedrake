"""
This test verifies that the Interior Penalty (IP) method for viscosity
converges at the expected order.

The viscosity term is of the form: div(nu [(grad u)+(grad u)^T])
with variable viscosity. Convergence is tested for RT2, DG1, BDM1 and BDM2
vector fields.
"""
import pytest
from firedrake import *
import numpy
import math


def outer_jump(v, n):
    """Jump function that represent the grad of a vector field."""
    return outer(v('+'), n('+'))+outer(v('-'), n('-'))


def run_test(family, degree, n):
    mesh = UnitSquareMesh(8*2**n, 8*2**n)
    x = mesh.coordinates
    k = 0.5*math.pi
    l = math.pi
    # made up analytical solution and viscosity:
    u0 = as_vector((sin(k*x[0])*cos(l*x[1]), cos(k*x[0])*sin(l*x[1])))
    nu = (1.+x[0]*(1.0-x[0]))*(1.+pow(x[1], 2))

    source = as_vector((
        -2*k*k*(pow(x[1], 2) + 1.0)*(x[0]*(x[0] - 1.0) - 1.0)
        * sin(k*x[0])*cos(l*x[1])
        + 2*k*(2*x[0] - 1.0)*(pow(x[1], 2) + 1.0)*cos(k*x[0])*cos(l*x[1])
        - l*(k + l)*(pow(x[1], 2) + 1.0)*(x[0]*(x[0] - 1.0) - 1.0)*sin(k*x[0])
        * cos(l*x[1])
        - 2*x[1]*(k + l)*(x[0]*(x[0] - 1.0) - 1.0)*sin(k*x[0])*sin(l*x[1]),
        -k*(k + l)*(pow(x[1], 2) + 1.0)*(x[0]*(x[0] - 1.0) - 1.0)*sin(l*x[1])
        * cos(k*x[0])
        - 2*l*l*(pow(x[1], 2) + 1.0)*(x[0]*(x[0] - 1.0) - 1.0)*sin(l*x[1])
        * cos(k*x[0])
        + 4*l*x[1]*(x[0]*(x[0] - 1.0) - 1.0)*cos(k*x[0])*cos(l*x[1])
        - (k + l)*(2*x[0] - 1.0)*(pow(x[1], 2) + 1.0)*sin(k*x[0])*sin(l*x[1])
    ))

    # grad u0 + (grad u0)^T :
    stress = as_tensor(((
        2*k*cos(k*x[0])*cos(l*x[1]),
        -(k + l)*sin(k*x[0])*sin(l*x[1])), (
        -(k + l)*sin(k*x[0])*sin(l*x[1]),
        2*l*cos(k*x[0])*cos(l*x[1]))))

    if family == "RT" or family == "BDM":
        V = FunctionSpace(mesh, family, degree)
    else:
        V = VectorFunctionSpace(mesh, family, degree)
    u = Function(V)
    v = TestFunction(V)

    # from Epshteyn et al. 2007 (http://dx.doi.org/10.1016/j.cam.2006.08.029)
    # the scheme is stable for alpha > 3*X*p*(p+1)*cot(theta), where X is the
    # maximum ratio of viscosity within a triangle, p the degree, and theta
    # the minimum angle in the mesh - we choose it a little higher here so we
    # get optimal convergence already for coarse meshes
    alpha = 25.*(degree+1)*degree
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    ds_Dir = ds((1, 2))  # Dirichlet boundary
    ds_Neu = ds((3, 4))  # Neumann boundary (enforces stress)

    # the IP viscosity term:
    F = (
        + inner(grad(v), nu*2.*sym(grad(u)))*dx
        + alpha/avg(h)*inner(outer_jump(v, n), nu*2.*sym(outer_jump(u, n)))*dS
        - inner(avg(grad(v)), nu*2.*sym(outer_jump(u, n)))*dS
        - inner(outer_jump(v, n), nu*2.*sym(avg(grad(u))))*dS
        + 2.0*alpha/h*inner(outer(v, n), nu*2.*sym(outer(u-u0, n)))*ds_Dir
        - inner(grad(v), nu*2.*sym(outer(u-u0, n)))*ds_Dir
        - inner(outer(v, n), nu*2.*sym(grad(u)))*ds_Dir
        - inner(outer(v, n), nu*stress)*ds_Neu
    )

    # the MMS source term:
    F += -inner(v, source)*dx

    solver_parameters = {'ksp_converged_reason': True,
                         'ksp_type': 'preonly',
                         'pc_type': 'lu'}
    solve(F == 0, u, solver_parameters=solver_parameters, nest=True)

    return norm(project(u-u0, V))


@pytest.mark.parametrize(('space'),
                         [("RT", 2), ("DG", 1), ("BDM", 1), ("BDM", 2)])
def test_ip_viscosity(space):
    family, degree = space
    errs = numpy.array([run_test(family, degree, n) for n in range(4)])
    orders = numpy.log(errs[:-1]/errs[1:])/numpy.log(2)
    print orders
    if family == "RT":
        expected_order = degree
    else:
        expected_order = degree+1.
    assert all(orders > 0.96*expected_order)


def test_indexed_interior_facet_gradients():
    """This is a regression test against a bug in coffee in combination with
    non-affine support, where it would evaluate some of the interior facet
    integrals from the IP scheme incorrectly. This seemed to only occur in
    an implementation where the terms are written using explicit summations
    over indices (as opposed to the tensorial notation above)."""
    mesh2d = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh2d, "RT", 1)
    uv = Function(V)
    U_test = TestFunction(V)

    n = FacetNormal(mesh2d)
    F1 = dot(jump(U_test[0], n), jump(uv[0], n))*dS
    F2 = -dot(avg(grad(U_test[0])), jump(uv[0], n))*dS
    F = F1+F2
    # the same thing written slightly differently:
    F0 = dot(jump(U_test[0], n)-avg(grad(U_test[0])), jump(uv[0], n))*dS

    M = assemble(derivative(F, uv)).M.values
    M1 = assemble(derivative(F1, uv)).M.values
    M2 = assemble(derivative(F2, uv)).M.values
    M0 = assemble(derivative(F0, uv)).M.values
    err = numpy.abs(M-(M1+M2)).max()
    print err
    assert(err < 1e-12)
    err = numpy.abs(M-M0).max()
    print err
    assert(err < 1e-12)


@pytest.mark.parametrize(('space'),
                         [("RT", 1), ("RT", 2), ("DG", 1), ("BDM", 1), ("BDM", 2)])
def test_stress_form_ip_penalty_term(space):
    """This is a regression test for a coffee issue with the alpha penalty in
    the IP viscosity term when using the full div(nu*sym(grad(u))) form of
    viscosity.  This term occurs in the mms test above as well, but the
    nonlinear solves seem to converge despite the wrong jacobian (on branches
    where this test fails)."""
    mesh2d = UnitSquareMesh(1, 1)
    family, degree = space
    if family == "RT" or family == "BDM":
        U = FunctionSpace(mesh2d, family, degree)
    else:
        U = VectorFunctionSpace(mesh2d, family, degree)
    v = TestFunction(U)
    u = Function(U)
    n = FacetNormal(mesh2d)

    F1 = inner(outer_jump(v, n), outer_jump(u, n))*dS
    F2 = inner(outer_jump(v, n), outer_jump(n, u))*dS
    F = inner(outer_jump(v, n), outer_jump(u, n)+outer_jump(n, u))*dS

    M1 = assemble(derivative(F1, u)).M.values
    M2 = assemble(derivative(F2, u)).M.values
    Ms = assemble(derivative(F1+F2, u)).M.values
    M = assemble(derivative(F, u)).M.values

    err = numpy.abs(M-(M1+M2)).max()
    assert(err < 1e-12)
    err = numpy.abs(M-Ms).max()
    assert(err < 1e-12)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
