from firedrake import *


def test_matfree_A_does_not_offload():
    length = 10
    n = 3
    mesh = RectangleMesh(2**n, 2**n, length, 1)

    P1 = FiniteElement("CG", cell=mesh.ufl_cell(), degree=1)
    B = FiniteElement("B", cell=mesh.ufl_cell(), degree=3)
    mini = P1 + B
    V = VectorFunctionSpace(mesh, mini)
    P = FunctionSpace(mesh, 'CG', 1)

    W = V*P

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    a = inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx + inner(div(u), q) * dx

    f = Constant((0, 0))
    L = inner(f, v) * dx

    # No-slip velocity boundary condition on top and bottom,
    # y == 0 and y == 1
    noslip = Constant((0, 0))
    bc0 = DirichletBC(W[0], noslip, (3, 4))

    # Parabolic inflow y(1-y) at x = 0 in positive x direction
    x = SpatialCoordinate(W.mesh())
    inflow = as_vector((x[1]*(1 - x[1]), 0.0))
    bc1 = DirichletBC(W[0], inflow, 1)

    # Zero pressure at outflow at x = 1
    bc2 = DirichletBC(W[1], 0.0, 2)

    bcs = [bc0, bc1, bc2]

    w = Function(W)

    u, p = w.subfunctions

    iterative_stokes_solver_parameters = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_type": "full",
        "fieldsplit_0": {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled": {
                "pc_type": "python",
                "pc_python_type": "firedrake.OffloadPC",
                "offload": {
                    "pc_type": "none",
                    "ksp_type": "preonly",
                }
            }
        },
        "fieldsplit_1": {
            "ksp_type": "preonly",
            "pc_type": "none",
        },
    }

    problem = LinearVariationalProblem(a, L, w, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=iterative_stokes_solver_parameters)
    solver.solve()

    ksp0 = solver.snes.ksp.pc.getFieldSplitSchurGetSubKSP()[0]
    assembled_ctx = ksp0.pc.getPythonContext()
    offload_ctx = assembled_ctx.pc.getPythonContext()
    A, P = offload_ctx.pc.getOperators()
    assert P.type == "seqaijcusparse"
    assert A.type == "python"
