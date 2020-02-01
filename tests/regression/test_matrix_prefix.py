from firedrake import *
import pytest


@pytest.mark.parametrize("options_prefix",
                         [None,
                          "",
                          "foo"])
def test_matrix_prefix_solver(options_prefix):
    parameters = {"ksp_type": "preonly",
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps",
                  "mat_mumps_icntl_14": 200,
                  "mat_mumps_icntl_24": 1}
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "P", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx
    L = v*dx
    uh = Function(V)

    problem = LinearVariationalProblem(a, L, uh)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters,
                                     options_prefix=options_prefix)
    solver.solve()

    pc = solver.snes.ksp.pc
    factor = pc.getFactorMatrix()
    assert factor.getType() == "mumps"
    assert factor.getMumpsIcntl(24) == 1

    for A in pc.getOperators():
        pfx = A.getOptionsPrefix()
        if pfx is None:
            pfx = ""
        assert pfx == solver.options_prefix


@pytest.mark.parametrize("options_prefix",
                         [None,
                          "",
                          "foo"])
def test_matrix_prefix_solver_assembled_pc(options_prefix):
    parameters = {
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled": {
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
            "mat_mumps_icntl_24": 1
        }
    }
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "P", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx
    L = v*dx
    uh = Function(V)

    problem = LinearVariationalProblem(a, L, uh)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters,
                                     options_prefix=options_prefix)
    solver.solve()

    pc = solver.snes.ksp.pc
    assert pc.getType() == "python"
    python = pc.getPythonContext()
    assert isinstance(python, AssembledPC)
    assembled = python.pc
    factor = assembled.getFactorMatrix()
    assert factor.getType() == "mumps"
    assert factor.getMumpsIcntl(24) == 1
