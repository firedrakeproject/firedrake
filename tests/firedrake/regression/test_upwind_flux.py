r"""
This code does the following.

First, obtains an upwind DG0 approximation to div(u*D).
Then, tries to find a BDM1 flux F such that div F is
equal to this upwind approximation.

we have

\int_e phi D_1 dx = -\int_e grad phi . u D dx
                    + \int_{\partial e} phi u.n \tilde{D} ds

where \tilde{D} is the value of D on the upwind face. For
DG0, grad phi = 0.

Then, if we define F such that

\int_f phi F.n ds = \int_f phi u.n \tilde{D} ds

then

\int_e phi div(F) ds = \int_{\partial e} phi u.n \tilde{D} ds
as required.
"""
from firedrake import *
import pytest


def run_test(quadrilateral):
    if quadrilateral:
        mesh = UnitCubedSphereMesh(refinement_level=2)
        RT_elt = FiniteElement("RTCF", "quadrilateral", 1)
    else:
        mesh = UnitIcosahedralSphereMesh(refinement_level=2)
        RT_elt = FiniteElement("RT", "triangle", 1)

    x = SpatialCoordinate(mesh)

    global_normal = as_vector((x[0]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]),
                               x[1]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]),
                               x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])))
    mesh.init_cell_orientations(global_normal)

    # Define function spaces and basis functions
    V_dg = FunctionSpace(mesh, "DG", 0)
    M = FunctionSpace(mesh, RT_elt)

    # advecting velocity
    u0 = as_vector((-x[1], x[0], 0))
    u = Function(M).project(u0)

    # Mesh-related functions
    n = FacetNormal(mesh)

    # ( dot(v, n) + |dot(v, n)| )/2.0
    un = 0.5*(dot(u, n) + abs(dot(u, n)))

    # D advection equation
    phi = TestFunction(V_dg)
    D = TrialFunction(V_dg)
    a_mass = inner(D, phi) * dx
    a_int = inner(-u*D, grad(phi)) * dx
    a_flux = inner(un('+')*D('+') - un('-')*D('-'), jump(phi)) * dS

    arhs = (a_int + a_flux)

    D1 = Function(V_dg)

    D0 = exp(-pow(x[2], 2) - pow(x[1], 2))
    D = Function(V_dg).interpolate(D0)

    D1problem = LinearVariationalProblem(a_mass, action(arhs, D), D1)
    D1solver = LinearVariationalSolver(D1problem)
    D1solver.solve()

    # Surface Flux equation
    V1 = FunctionSpace(mesh, RT_elt)
    w = TestFunction(V1)
    Ft = TrialFunction(V1)
    Fs = Function(V1)

    aFs = (inner(n('+'), w('+')) * inner(Ft('+'), n('+'))
           + inner(n('-'), w('-')) * inner(Ft('-'), n('-'))) * dS
    LFs = 2.0*(inner(n('+'), w('+')) * un('+') * D('+')
               + inner(n('-'), w('-')) * un('-') * D('-')) * dS

    Fsproblem = LinearVariationalProblem(aFs, LFs, Fs)
    Fssolver = LinearVariationalSolver(Fsproblem,
                                       solver_parameters={'ksp_type': 'preonly'})
    Fssolver.solve()

    divFs = Function(V_dg)

    solve(a_mass == inner(div(Fs), phi) * dx, divFs)

    assert errornorm(divFs, D1, degree_rise=0) < 1e-12


def test_upwind_flux_icosahedral_sphere():
    run_test(quadrilateral=False)


@pytest.mark.parallel
def test_upwind_flux_icosahedral_sphere_parallel():
    run_test(quadrilateral=False)


def test_upwind_flux_cubed_sphere():
    run_test(quadrilateral=True)


@pytest.mark.parallel
def test_upwind_flux_cubed_sphere_parallel():
    run_test(quadrilateral=True)
