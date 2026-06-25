from firedrake import *
import numpy as np


def run_stokes_mini(n):
    length = 10
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
    bc0 = [DirichletBC(W[0], noslip, (3, 4))]

    # Parabolic inflow y(1-y) at x = 0 in positive x direction
    x = SpatialCoordinate(W.mesh())
    inflow = as_vector((x[1]*(1 - x[1]), 0.0))
    bc1 = DirichletBC(W[0], inflow, 1)

    # Zero pressure at outlow at x = 1
    bc2 = DirichletBC(W[1], 0.0, 2)

    bcs = bc0 + [bc1, bc2]

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
                    "pc_type": "ksp",
                    "ksp": {
                        "ksp_monitor_true_residual": None,
                        "ksp_converged_reason": None,
                        "ksp_type": "cg",
                        "ksp_rtol": 1e-5,
                        "ksp_max_it": 1000,
                    }
                }
            }
        },
        "fieldsplit_1": {
            "pc_type": "none",
        },
    }

    problem = LinearVariationalProblem(a, L, w, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=iterative_stokes_solver_parameters)
    solver.solve()

    # ksp0 = solver.snes.ksp.pc.getFieldSplitSchurGetSubKSP()[0]
    # sub_pc = ksp0.pc.???
    # offload_python_context = sub_pc.getPythonContext()
    # assert offload_python_context.pc.P.type == "seqaijcusparse"
    # assert offload_python_context.pc.A.type == "python"

    # We've set up Poiseuille flow, so we expect a parabolic velocity
    # field and a linearly decreasing pressure.
    x, y = SpatialCoordinate(mesh)
    uexact = as_vector([y*(1 - y), 0])
    pexact = 2*(length - x)

    return errornorm(uexact, u, degree_rise=0), errornorm(pexact, p, degree_rise=0)


def test_no_offload_A():
    u_err = []
    p_err = []

    for n in range(3, 6):
        errs = run_stokes_mini(n)
        u_err.append(errs[0])
        p_err.append(errs[1])

    u_err = np.asarray(u_err)
    p_err = np.asarray(p_err)
    assert (np.log2(u_err[:-1] / u_err[1:]) > 2).all()
    assert (np.log2(p_err[:-1] / p_err[1:]) > 1.5).all()
