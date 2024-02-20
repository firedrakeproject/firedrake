import pytest
from firedrake import *
import numpy as np


@pytest.fixture(scope="module",
                params=[0, 1, 2, 3],
                ids=lambda x: "DG%d" % x)
def degree(request):
    return request.param


@pytest.fixture(scope="module")
def threshold(degree):
    return {0: 0.98,
            1: 1.8,
            2: 2.9,
            3: 3.9}[degree]


def run_test(degree):
    l2error = []
    # Advect a sine wave with a constant, unit velocity for 200
    # timesteps (dt = 5e-5)
    for n in range(6, 10):
        mesh = PeriodicUnitIntervalMesh(2**n)
        x = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, 'DG', degree)
        u = Constant((1, ))
        D = TrialFunction(V)
        phi = TestFunction(V)
        n = FacetNormal(mesh)
        un = 0.5 * (dot(u, n) + abs(dot(u, n)))

        a_mass = phi*D*dx
        a_int = dot(grad(phi), -u*D)*dx
        a_flux = dot(jump(phi), jump(un*D))*dS

        dD1 = Function(V)
        D1 = Function(V)

        t = Constant(0)
        exact = sin(2*pi*(x[0] - t))
        D = Function(V).interpolate(exact)

        nstep = 200
        dt = Constant(5e-5)
        arhs = action(a_mass - dt * (a_int + a_flux), D1)
        rhs = Cofunction(V.dual())

        # Since DG mass-matrix is diagonal, just assemble the
        # diagonal and then "solve" is an entry-wise division.
        mass_diag = assemble(a_mass, diagonal=True)

        def solve(mass_diag, arhs, rhs, update):
            with assemble(arhs, tensor=rhs).dat.vec_ro as x:
                with update.dat.vec as y, mass_diag.dat.vec_ro as d:
                    y.pointwiseDivide(x, d)

        for _ in range(nstep):
            # SSPRK3
            D1.assign(D)
            solve(mass_diag, arhs, rhs, dD1)

            D1.assign(dD1)
            solve(mass_diag, arhs, rhs, dD1)

            D1.assign(0.75*D + 0.25*dD1)
            solve(mass_diag, arhs, rhs, dD1)
            D.assign((1.0/3.0)*D + (2.0/3.0)*dD1)

        D1.assign(D)

        t.assign(float(dt) * nstep)

        D.interpolate(exact)

        l2error.append(norm(assemble(D1 - D)))

    return np.asarray(l2error)


@pytest.mark.skipcomplexnoslate
def test_periodic_1d_advection(degree, threshold):
    l2error = run_test(degree)
    convergence = np.log2(l2error[:-1] / l2error[1:])

    assert np.all(convergence > threshold)


@pytest.mark.skipcomplexnoslate
@pytest.mark.parallel(nprocs=2)
def test_periodic_1d_advection_parallel(degree, threshold):
    l2error = run_test(degree)
    convergence = np.log2(l2error[:-1] / l2error[1:])

    assert np.all(convergence > threshold)
