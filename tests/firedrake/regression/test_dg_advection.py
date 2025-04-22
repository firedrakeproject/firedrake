from firedrake import *
import numpy as np
import pytest


def run_test(mesh):
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    V = FunctionSpace(mesh, "DG", 0)
    M = VectorFunctionSpace(mesh, "CG", 1)

    # advecting velocity
    u0 = as_vector((-x[1]*(1 - x[2]*x[2]), x[0]*(1 - x[2]*x[2]), Constant(0)))
    u = Function(M).interpolate(u0)

    dt = (pi/3) * 0.006
    Dt = Constant(dt)

    phi = TestFunction(V)
    D = TrialFunction(V)

    n = FacetNormal(mesh)

    un = 0.5 * (dot(u, n) + abs(dot(u, n)))

    a_mass = inner(D, phi) * dx
    a_int = inner(-u*D, grad(phi))*dx
    a_flux = inner(un('+')*D('+') - un('-')*D('-'), jump(phi))*dS
    arhs = a_mass - Dt * (a_int + a_flux)

    dD1 = Function(V)
    D1 = Function(V)

    D0 = conditional(le(real(x[0]), 0), 1, 0.0)
    D = Function(V).interpolate(D0)

    t = 0.0
    T = 10*dt

    problem = LinearVariationalProblem(a_mass, action(arhs, D1), dD1,
                                       constant_jacobian=True)
    solver = LinearVariationalSolver(
        problem,
        solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        }
    )

    L2_0 = norm(D)
    Dbar_0 = assemble(D*dx)
    while t < (T - dt/2):
        D1.assign(D)
        solver.solve()
        D1.assign(dD1)

        solver.solve()
        D1.assign(0.75*D + 0.25*dD1)
        solver.solve()
        D.assign((1.0/3.0)*D + (2.0/3.0)*dD1)

        t += dt

    L2_T = norm(D)
    Dbar_T = assemble(D*dx)

    # L2 norm decreases
    assert L2_T < L2_0

    # Mass conserved
    assert np.allclose(Dbar_T, Dbar_0)


@pytest.mark.parallel([1, 3])
def test_dg_advection_icosahedral_sphere():
    run_test(UnitIcosahedralSphereMesh(refinement_level=3))


@pytest.mark.parallel([1, 3])
def test_dg_advection_cubed_sphere():
    run_test(UnitCubedSphereMesh(refinement_level=4))
