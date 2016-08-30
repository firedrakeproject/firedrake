import pytest
from firedrake import *
import numpy as np


def run_advection(degree):
    l2error = []
    nx = lambda n: 2**n
    u = Constant((1, 0))
    exact_expr = "exp(-(x[0] - 0.5 -t )*(x[0] - 0.5 -t)*100.0)"

    # Advect a sine wave with a constant, unit velocity for 200
    # timesteps (dt = 5e-5)
    for n in range(7, 9):
        print n
        # mesh = PeriodicIntervalMesh(nx(n), 1)
        mesh = PeriodicUnitSquareMesh(nx(n), nx(n))
        V = FunctionSpace(mesh, "DG", degree)
        D = TrialFunction(V)
        phi = TestFunction(V)
        n = FacetNormal(mesh)
        un = 0.5 * (dot(u, n) + abs(dot(u, n)))

        a_mass = phi*D*dx
        a_int = dot(grad(phi), -u*D)*dx
        a_flux = dot(jump(phi), jump(un*D))*dS

        dD1 = Function(V)
        D1 = Function(V)
        exact = Expression(exact_expr, t=0)

        D = Function(V).interpolate(exact)

        nstep = 200
        dt = Constant(5e-5)
        arhs = action(a_mass - dt * (a_int + a_flux), D1)
        rhs = Function(V)

        # Since DG mass-matrix is block diagonal, just assemble the
        # inverse and then "solve" is a matvec.
        mass_inv = assemble(a_mass, inverse=True).M.handle

        def solve(mass_inv, arhs, rhs, update):
            with assemble(arhs, tensor=rhs).dat.vec_ro as x:
                with update.dat.vec as y:
                    mass_inv.mult(x, y)

        limiter = KuzminLimiter(V)
        for _ in range(nstep):
            # SSPRK3
            D1.assign(D)
            limiter.apply(D1)
            solve(mass_inv, arhs, rhs, dD1)

            D1.assign(dD1)
            limiter.apply(D1)
            solve(mass_inv, arhs, rhs, dD1)

            D1.assign(0.75*D + 0.25*dD1)
            limiter.apply(D1)
            solve(mass_inv, arhs, rhs, dD1)
            D.assign((1.0/3.0)*D + (2.0/3.0)*dD1)
            limiter.apply(D)

        D1.assign(D)

        exact.t = float(dt) * nstep

        D.interpolate(exact)

        D.rename("exact")
        D1.rename("computed")
        diff = Function(V, name="diff")
        diff.assign(D - D1)
        # File("output.pvd").write(D, D1, diff)
        l2error.append(norm(assemble(D1 - D)))

    return np.asarray(l2error)


errors = run_advection(2)

print errors
print np.log2(errors[:-1]/errors[1:])
