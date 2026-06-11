from firedrake import *
import numpy as np

def solve_poisson():
    mesh = RectangleMesh(10, 10, 2.0, 1.0)
    x, y = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    u = Function(V)
    solve(a == L, u)
    return u.dat.data_ro

def test_coupling():
    mesh_1 = UnitSquareMesh(10, 10)
    mesh_1.coordinates.dat.data[:, 0] += 1.0  # Shift to the right by 1
    mesh_2 = UnitSquareMesh(10, 10)

    x1, y1 = SpatialCoordinate(mesh_1)

    V1 = FunctionSpace(mesh_1, "CG", 1)
    V2 = FunctionSpace(mesh_2, "CG", 1)
    W = V1 * V2

    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    # RHS functions
    f = Function(W)
    f1, f2 = f.subfunctions

    # Nitsche penalty weights
    w1 = Constant(100.0)
    w2 = Constant(100.0)

    n1 = FacetNormal(mesh_1)
    n2 = FacetNormal(mesh_2)

    dx1 = Measure("dx", domain=mesh_1)
    dx2 = Measure("dx", domain=mesh_2)
    ds1 = Measure("ds", domain=mesh_1, subdomain_id=1)
    ds2 = Measure("ds", domain=mesh_2, subdomain_id=2)

    # Poisson
    A11_form = inner(grad(u1), grad(v1)) * dx1 \
                - inner(dot(grad(u1), n1), v1) * ds1 \
                + w1 * inner(u1, v1) * ds1
    # Poisson
    A22_form = inner(grad(u2), grad(v2)) * dx2 \
            - inner(dot(grad(u2), n2), v2) * ds2 \
            + w2 * inner(u2, v2) * ds2

    # A12: row v1, column u2
    A12_form = -w1 * inner(interpolate(u2, V1, allow_missing_dofs=True), v1) * ds1

    # A21: row v2, column u1
    A21_form = -w2 * inner(interpolate(u1, V2, allow_missing_dofs=True), v2) * ds2

    # Linear form
    b1 = inner(f1, v1) * dx1
    b2 = inner(f2, v2) * dx2

    bc = DirichletBC(W.sub(1), 0, [1, 3, 4])

    A = A11_form + A12_form + A21_form + A22_form
    L = b1 + b2

    u_sol = Function(W)
    problem = LinearVariationalProblem(A, L, u_sol, bcs=bc)
    solver = LinearVariationalSolver(problem)
    solver.solve()

    u1_sol, u2_sol = u_sol.subfunctions
    u1_sol_data = np.concatenate((u1_sol.dat.data_ro, u2_sol.dat.data_ro))
    assert np.allclose(u1_sol_data, solve_poisson(), atol=1e-2)

