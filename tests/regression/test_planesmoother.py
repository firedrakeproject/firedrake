import pytest
import numpy
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


@pytest.mark.skipcomplex
def test_xy_equivalence():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(Constant(1), v)*dx
    bcs = DirichletBC(V, Constant(0), "on_boundary")
    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    patch_defined = LinearVariationalSolver(problem,
                                            options_prefix="",
                                            solver_parameters={"mat_type": "matfree",
                                                               "ksp_type": "cg",
                                                               "pc_type": "python",
                                                               "pc_python_type": "firedrake.PatchPC",
                                                               "patch_pc_patch_construct_type": "python",
                                                               "patch_pc_patch_construct_python_type": "firedrake.PlaneSmoother",
                                                               "patch_pc_patch_construct_ps_sweeps": "0+10:1+10",
                                                               "patch_sub_ksp_type": "preonly",
                                                               "patch_sub_pc_type": "lu",
                                                               "ksp_monitor": None})

    patch_defined.snes.ksp.setConvergenceHistory()
    patch_defined.solve()
    patch_defined_history = patch_defined.snes.ksp.getConvergenceHistory()

    appctx = {}
    appctx["my_x"] = lambda z: z[0]
    appctx["my_y"] = lambda z: z[1]
    user_defined = LinearVariationalSolver(problem,
                                           options_prefix="",
                                           solver_parameters={"mat_type": "matfree",
                                                              "ksp_type": "cg",
                                                              "pc_type": "python",
                                                              "pc_python_type": "firedrake.PatchPC",
                                                              "patch_pc_patch_construct_type": "python",
                                                              "patch_pc_patch_construct_python_type": "firedrake.PlaneSmoother",
                                                              "patch_pc_patch_construct_ps_sweeps": "my_x+10:my_y+10",
                                                              "patch_sub_ksp_type": "preonly",
                                                              "patch_sub_pc_type": "lu",
                                                              "ksp_monitor": None},
                                           appctx=appctx)

    user_defined.snes.ksp.setConvergenceHistory()
    uh.assign(0)
    user_defined.solve()
    user_defined_history = user_defined.snes.ksp.getConvergenceHistory()

    assert numpy.allclose(patch_defined_history, user_defined_history)


@pytest.mark.skipcomplex
def test_divisions_equivalence():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(Constant(1), v)*dx
    bcs = DirichletBC(V, Constant(0), "on_boundary")
    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    patch_defined = LinearVariationalSolver(problem,
                                            options_prefix="",
                                            solver_parameters={"mat_type": "matfree",
                                                               "ksp_type": "cg",
                                                               "pc_type": "python",
                                                               "pc_python_type": "firedrake.PatchPC",
                                                               "patch_pc_patch_construct_type": "python",
                                                               "patch_pc_patch_construct_python_type": "firedrake.PlaneSmoother",
                                                               "patch_pc_patch_construct_ps_sweeps": "0+10:1+10",
                                                               "patch_sub_ksp_type": "preonly",
                                                               "patch_sub_pc_type": "lu",
                                                               "ksp_monitor": None})

    patch_defined.snes.ksp.setConvergenceHistory()
    patch_defined.solve()
    patch_defined_history = patch_defined.snes.ksp.getConvergenceHistory()

    appctx = {}
    appctx["x_div"] = numpy.linspace(0.0, 1.0, 11)
    appctx["y_div"] = numpy.linspace(0.0, 1.0, 11)
    user_defined = LinearVariationalSolver(problem,
                                           options_prefix="",
                                           solver_parameters={"mat_type": "matfree",
                                                              "ksp_type": "cg",
                                                              "pc_type": "python",
                                                              "pc_python_type": "firedrake.PatchPC",
                                                              "patch_pc_patch_construct_type": "python",
                                                              "patch_pc_patch_construct_python_type": "firedrake.PlaneSmoother",
                                                              "patch_pc_patch_construct_ps_sweeps": "0+x_div:1+y_div",
                                                              "patch_sub_ksp_type": "preonly",
                                                              "patch_sub_pc_type": "lu",
                                                              "ksp_monitor": None},
                                           appctx=appctx)

    user_defined.snes.ksp.setConvergenceHistory()
    uh.assign(0)
    user_defined.solve()
    user_defined_history = user_defined.snes.ksp.getConvergenceHistory()

    assert numpy.allclose(patch_defined_history, user_defined_history)


@pytest.mark.skipcomplex
def test_tensor_grids():
    x_points = numpy.logspace(0, 1, 10)
    y_points = numpy.array([0, 0.2, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.49, 1.5])
    mesh = TensorRectangleMesh(x_points, y_points)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(Constant(1), v)*dx
    bcs = DirichletBC(V, Constant(0), "on_boundary")
    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    appctx = {}
    appctx["x_div"] = 0.5*(x_points[:-1]+x_points[1:])
    appctx["y_div"] = 0.5*(y_points[:-1]+y_points[1:])
    user_defined = LinearVariationalSolver(problem,
                                           options_prefix="",
                                           solver_parameters={"mat_type": "matfree",
                                                              "ksp_type": "cg",
                                                              "pc_type": "python",
                                                              "pc_python_type": "firedrake.PatchPC",
                                                              "patch_pc_patch_construct_type": "python",
                                                              "patch_pc_patch_construct_python_type": "firedrake.PlaneSmoother",
                                                              "patch_pc_patch_construct_ps_sweeps": "0+x_div:1+y_div",
                                                              "patch_sub_ksp_type": "preonly",
                                                              "patch_sub_pc_type": "lu",
                                                              "ksp_monitor": None},
                                           appctx=appctx)

    user_defined.solve()
    nits = user_defined.snes.getLinearSolveIterations()

    assert nits == 15


@pytest.mark.skipcomplex
def test_not_aligned():
    baseN = 4
    nrefs = 2
    base = UnitSquareMesh(baseN, baseN)
    mh = MeshHierarchy(base, nrefs)
    mesh = mh[-1]
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(Constant(1), v)*dx
    bcs = DirichletBC(V, Constant(0), "on_boundary")
    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    solver_parameters = {"mat_type": "matfree",
                         "ksp_type": "cg",
                         "pc_type": "mg",
                         "pc_mg_cycle_type": "v",
                         "pc_mg_type": "multiplicative",
                         "mg_levels_ksp_type": "chebyshev",
                         "mg_levels_ksp_max_it": 2,
                         "mg_levels_pc_type": "python",
                         "mg_levels_pc_python_type": "firedrake.PatchPC",
                         "mg_levels_patch_pc_patch_construct_type": "python",
                         "mg_levels_patch_pc_patch_construct_python_type": "firedrake.PlaneSmoother",
                         "mg_levels_patch_sub_ksp_type": "preonly",
                         "mg_levels_patch_sub_pc_type": "lu",
                         "mg_coarse_pc_type": "python",
                         "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                         "mg_coarse_assembled_pc_type": "lu",
                         "mg_coarse_assembled_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER,
                         "ksp_monitor": None}
    appctx = {}
    appctx["x_plus_y"] = lambda z: z[0]+z[1]
    appctx["x_minus_y"] = lambda z: z[0] - z[1]
    N = baseN
    for j in range(nrefs):
        N *= 2
        appctx["x_plus_y_divisions"+str(j+1)] = numpy.linspace(0.0, 2.0, 2*N+1)
        appctx["x_minus_y_divisions"+str(j+1)] = numpy.linspace(-1.0, 1.0, 2*N+1)
        solver_parameters["mg_levels_"+str(j+1)+"_patch_pc_patch_construct_ps_sweeps"] = "x_plus_y+x_plus_y_divisions"+str(j+1)+":x_minus_y+x_minus_y_divisions"+str(j+1)

    user_defined = LinearVariationalSolver(problem,
                                           options_prefix="",
                                           solver_parameters=solver_parameters,
                                           appctx=appctx)

    user_defined.solve()
    nits = user_defined.snes.getLinearSolveIterations()

    assert nits == 7
