from firedrake import *
import numpy as np


def build_limiter(space):
    taylor_space1 = FunctionSpace(space.mesh(), "Discontinuous Taylor", 1)
    taylor_space2 = FunctionSpace(space.mesh(), "Discontinuous Taylor", 2)
    pdg1 = FunctionSpace(space.mesh(), "DG", 1)
    pdg2 = space
    cg1 = FunctionSpace(space.mesh(), "CG", 1)

    min_field = Function(cg1)
    max_field = Function(cg1)
    taylor_field1 = Function(taylor_space1)
    taylor_field2 = Function(taylor_space2)

    def compute_centers(field):
        taylor_field2.project(field)
        min_field.assign(1e10)
        max_field.assign(-1e10)
        par_loop("""
        for(int i = 0; i < min_field.dofs; i++) {
            min_field[i][0] = fmin(min_field[i][0], t[0][0]*2);
            max_field[i][0] = fmax(max_field[i][0], t[0][0]*2);
            printf("%g\\n", max_field[i][0]);
            printf("%g\\n", min_field[i][0]);
        }
        """,dx,
                 {"t": (taylor_field2, READ),
                  "min_field": (min_field, RW),
                  "max_field": (max_field, RW)})

    def limit(field):
        compute_centers(field)
        np.max(taylor_field2.dat.data)

    return limit


def test_step_function_loop(iterations=70):
    mesh = PeriodicUnitSquareMesh(10, 10)
    degree = 2

    # test function space
    v = FunctionSpace(mesh, "DG", degree)
    m = VectorFunctionSpace(mesh, "CG", 1)

    # advecting velocity
    if m.shape == (1,):
        u0 = as_vector([1])
    else:
        u0 = as_vector([1, 0])
    u = Function(m).interpolate(u0)

    # advection problem
    dt = 1. / iterations
    phi = TestFunction(v)
    D = TrialFunction(v)
    n = FacetNormal(mesh)
    un = 0.5 * (dot(u, n) + abs(dot(u, n)))  # upwind value

    a_mass = phi * D * dx
    a_int = dot(grad(phi), -u * D) * dx
    a_flux = dot(jump(phi), un('+') * D('+') - un('-') * D('-')) * dS
    arhs = a_mass - dt * (a_int + a_flux)

    dD1 = Function(v)
    D1 = Function(v)
    x = SpatialCoordinate(mesh)

    # Initial Conditions
    D0 = conditional(x[0] > 0.5, 1, 0.)

    D = Function(v).interpolate(D0)
    D1.assign(D)
    D1_old = Function(D1)

    t = 0.0
    T = iterations * dt
    problem = LinearVariationalProblem(a_mass, action(arhs, D1), dD1)
    solver = LinearVariationalSolver(problem, solver_parameters={'ksp_type': 'cg'})

    ufile = File("test_limiter.pvd")
    ufile.write(D1)
    # Make slope limiter
    # limiter = KuzminLimiter(v)
    # limiter.apply(D1)
    limer = build_limiter(v)
    limer(D1)
    #
    # while t < (T - dt / 2):
    #     D1.assign(D)
    #     limiter.apply(D1)
    #     solver.solve()
    #     D1.assign(dD1)
    #     limiter.apply(D1)
    #
    #     solver.solve()
    #     D1.assign(0.75 * D + 0.25 * dD1)
    #     limiter.apply(D1)
    #     solver.solve()
    #     D.assign((1.0 / 3.0) * D + (2.0 / 3.0) * dD1)
    #     limiter.apply(D)
    #
    #     ufile.write(D1)
    #     t += dt
    #
    # diff = assemble((D1 - D1_old) ** 2 * dx) ** 0.5
    # print "Error:", diff
    # max = np.max(D1.dat.data_ro)
    # min = np.min(D1.dat.data_ro)
    # print "Max:", max, "Min:", min
    # assert max <= 1.0 + 1e-2, "Failed by exceeding max values"
    # assert max <= 1.0 + 1e-2, "Failed by exceeding max values"

if __name__ == '__main__':
    test_step_function_loop()
