from firedrake import *



def test_coupling():
    n = 16
    mesh_1 = RectangleMesh(n, n, 2.0, 1.0, originX=1.0, quadrilateral=True)
    mesh_2 = RectangleMesh(n, n, 1.0, 1.0)

    x1, y1 = SpatialCoordinate(mesh_1)
    x2, y2 = SpatialCoordinate(mesh_2)

    V1 = FunctionSpace(mesh_1, "CG", 2)
    V2 = FunctionSpace(mesh_2, "CG", 2)
    W = V1 * V2

    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    # Intermediate spaces for cross-mesh interpolation.
    Q1 = FunctionSpace(mesh_1, "CG", 2)
    Q2 = FunctionSpace(mesh_2, "CG", 2)

    exact_1 = x1**2 + y1**2
    exact_2 = x2**2 + y2**2
    f1 = -div(grad(exact_1))
    f2 = -div(grad(exact_2))

    # Nitsche penalty weights
    gamma = Constant(100.0)
    w1 = gamma / CellDiameter(mesh_1)
    w2 = gamma / CellDiameter(mesh_2)

    n1 = FacetNormal(mesh_1)
    n2 = FacetNormal(mesh_2)

    dx1 = Measure("dx", domain=mesh_1)
    dx2 = Measure("dx", domain=mesh_2)
    ds1 = Measure("ds", domain=mesh_1)
    ds2 = Measure("ds", domain=mesh_2)
    interface_1 = 1
    interface_2 = 2

    A11_form = inner(grad(u1), grad(v1)) * dx1 \
                - inner(dot(grad(u1), n1), v1) * ds1(interface_1) \
                - inner(dot(grad(v1), n1), u1) * ds1(interface_1) \
                + w1 * inner(u1, v1) * ds1(interface_1)
    A22_form = inner(grad(u2), grad(v2)) * dx2 \
            - inner(dot(grad(u2), n2), v2) * ds2(interface_2) \
            - inner(dot(grad(v2), n2), u2) * ds2(interface_2) \
            + w2 * inner(u2, v2) * ds2(interface_2)

    # A12: row v1, column u2
    q1 = TrialFunction(Q1)
    M1 = (inner(q1, inner(grad(v1), n1)) - w1 * inner(q1, v1)) * ds1(interface_1)  # Q1 -> W^*
    B12 = interpolate(u2, Q1, allow_missing_dofs=True)  # W -> Q1
    A12_form = action(M1, B12)  # W -> W^*

    # A21: row v2, column u1
    q2 = TrialFunction(Q2)
    M2 = (inner(q2, inner(grad(v2), n2)) - w2 * inner(q2, v2)) * ds2(interface_2)  # Q2 -> W^*
    B21 = interpolate(u1, Q2, allow_missing_dofs=True)  # W -> Q2
    A21_form = action(M2, B21)  # W -> W^*

    # Linear form
    b1 = inner(f1, v1) * dx1
    b2 = inner(f2, v2) * dx2

    bcs = [
        DirichletBC(W.sub(0), exact_1, [2, 3, 4]),
        DirichletBC(W.sub(1), exact_2, [1, 3, 4]),
    ]

    A = A11_form + A12_form + A21_form + A22_form
    L = b1 + b2

    u_sol = Function(W)
    problem = LinearVariationalProblem(A, L, u_sol, bcs=bcs)
    solver = LinearVariationalSolver(
        problem,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    solver.solve()

    u1_sol, u2_sol = u_sol.subfunctions
    assert errornorm(exact_1, u1_sol) < 1e-10
    assert errornorm(exact_2, u2_sol) < 1e-10


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
