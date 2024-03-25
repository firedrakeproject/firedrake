from firedrake import *
import numpy


def test_serendipity_biharmonic():

    sp = {"snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
          "mat_mumps_icntl_14": 200}

    def error(N):
        mesh = UnitSquareMesh(N, N, quadrilateral=True)
        degree = 2
        V = FunctionSpace(mesh, "S", degree)
        u = Function(V)
        v = TestFunction(V)

        X = SpatialCoordinate(mesh)
        k = 1
        u_ex = sin(2*pi*k*X[0]) * cos(2*pi*k*X[1])

        h = avg(CellDiameter(mesh))
        alpha = Constant(1)
        n = FacetNormal(mesh)

        f = div(div(grad(grad(u_ex)))) + u_ex
        g = dot(grad(grad(u_ex)), n)
        F = inner(grad(grad(u)), grad(grad(v)))*dx + inner(u, v)*dx - inner(f, v)*dx - inner(g, grad(v))*ds + alpha * h**(-(degree+1)) * inner(jump(grad(u), n), jump(grad(v), n))*dS

        bc = DirichletBC(V, project(u_ex, V, solver_parameters=sp), "on_boundary")

        solve(F == 0, u, bc, solver_parameters=sp)

        err = errornorm(u_ex, u, "L2")
        return err

    errors = []
    for N in [10, 20, 40]:
        errors.append(error(N))
    errors = numpy.array(errors)

    convergence_orders = lambda x: numpy.log2(x[:-1] / x[1:])
    conv = convergence_orders(errors)

    assert all(conv >= 2)
