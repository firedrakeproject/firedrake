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
        + inner(nu*2.*sym(grad(u)), grad(v)) * dx
        + alpha/avg(h)*inner(outer_jump(v, n), nu*2.*sym(outer_jump(u, n))) * dS
        - inner(nu*2.*sym(outer_jump(conj(u), n)), avg(grad(v))) * dS
        - inner(nu*2.*sym(avg(grad(u))), outer_jump(conj(v), n)) * dS
        + 2.0*alpha/h*inner(outer(v, n), nu*2.*sym(outer(u-u0, n))) * ds_Dir
        - inner(nu*2.*sym(outer(conj(u-u0), n)), grad(v)) * ds_Dir
        - inner(nu*2.*sym(grad(u)), outer(conj(v), n)) * ds_Dir
        - inner(outer(v, n), nu*stress) * ds_Neu
    )

    # the MMS source term:
    F += -inner(source, v) * dx

    solver_parameters = {'ksp_converged_reason': None,
                         'ksp_type': 'preonly',
                         'pc_type': 'lu',
                         'mat_type': 'nest'}
    solve(F == 0, u, solver_parameters=solver_parameters)

    return norm(project(u-u0, V))


@pytest.mark.parametrize(('space'),
                         [("RT", 2), ("DG", 1), ("BDM", 1), ("BDM", 2)])
def test_ip_viscosity(space):
    family, degree = space
    errs = numpy.array([run_test(family, degree, n) for n in range(4)])
    orders = numpy.log(errs[:-1]/errs[1:])/numpy.log(2)
    if family == "RT":
        expected_order = degree
    else:
        expected_order = degree+1.
    assert all(orders > 0.96*expected_order)
