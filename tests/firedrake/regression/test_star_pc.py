import pytest
import warnings
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


pytest.skip(allow_module_level=True, reason="pyop3 TODO")


@pytest.fixture(params=["scalar",
                        pytest.param("vector", marks=pytest.mark.skipcomplexnoslate),
                        pytest.param("mixed", marks=pytest.mark.skipcomplexnoslate)])
def problem_type(request):
    return request.param


@pytest.fixture(params=["petscasm", "tinyasm"])
def backend(request):
    return request.param


def filter_warnings(caller):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "Creating new TransferManager", RuntimeWarning)
        caller()


def test_star_equivalence(problem_type, backend):
    distribution_parameters = {"partition": True,
                               "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

    if problem_type == "scalar":
        base = UnitSquareMesh(10, 10, distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base, 2, distribution_parameters=distribution_parameters)
        mesh = mh[-1]
        V = FunctionSpace(mesh, "CG", 1)

        u = Function(V)
        v = TestFunction(V)

        a = inner(grad(u), grad(v))*dx - inner(Constant(1), v)*dx
        bcs = DirichletBC(V, 0, "on_boundary")
        nsp = None

        star_params = {"mat_type": "aij",
                       "snes_type": "ksponly",
                       "snes_view": None,
                       "ksp_type": "richardson",
                       "pc_type": "mg",
                       "pc_mg_type": "multiplicative",
                       "pc_mg_cycles": "v",
                       "mg_levels_ksp_type": "richardson",
                       "mg_levels_ksp_richardson_scale": 0.5,
                       "mg_levels_ksp_max_it": 1,
                       "mg_levels_pc_type": "python",
                       "mg_levels_pc_python_type": "firedrake.ASMStarPC",
                       "mg_levels_pc_star_construct_dim": 0,
                       "mg_levels_pc_star_view_patch_sizes": None}

        comp_params = {"mat_type": "aij",
                       "snes_type": "ksponly",
                       "ksp_type": "richardson",
                       "pc_type": "mg",
                       "pc_mg_type": "multiplicative",
                       "pc_mg_cycles": "v",
                       "mg_levels_ksp_type": "richardson",
                       "mg_levels_ksp_richardson_scale": 0.5,
                       "mg_levels_ksp_max_it": 1,
                       "mg_levels_pc_type": "jacobi"}

    elif problem_type == "vector":
        base = UnitCubeMesh(2, 2, 2, distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base, 2, distribution_parameters=distribution_parameters)
        mesh = mh[-1]
        V = FunctionSpace(mesh, "RT", 2)

        u = Function(V)
        v = TestFunction(V)

        (x, y, z) = SpatialCoordinate(mesh)
        f = as_vector([x*(1-x), y*(1-y), z*(1-z)])
        a = inner(div(u), div(v))*dx + inner(u, v)*dx - inner(f, v)*dx
        bcs = DirichletBC(V, 0, "on_boundary")
        nsp = None

        star_params = {"mat_type": "aij",
                       "snes_type": "ksponly",
                       "ksp_type": "cg",
                       "pc_type": "mg",
                       "pc_mg_type": "multiplicative",
                       "pc_mg_cycle_type": "v",
                       "mg_levels_ksp_type": "richardson",
                       "mg_levels_ksp_richardson_scale": 1/4,
                       "mg_levels_ksp_max_it": 1,
                       "mg_levels_pc_type": "python",
                       "mg_levels_pc_python_type": "firedrake.ASMStarPC",
                       "mg_levels_pc_star_construct_dim": 1,
                       "mg_coarse_pc_type": "lu",
                       "mg_coarse_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER}

        comp_params = {"mat_type": "aij",
                       "snes_type": "ksponly",
                       "ksp_type": "cg",
                       "pc_type": "mg",
                       "pc_mg_type": "multiplicative",
                       "pc_mg_cycle_type": "v",
                       "mg_levels_ksp_type": "richardson",
                       "mg_levels_ksp_richardson_scale": 1/4,
                       "mg_levels_ksp_max_it": 1,
                       "mg_levels_pc_type": "python",
                       "mg_levels_pc_python_type": "firedrake.PatchPC",
                       "mg_levels_patch_pc_patch_save_operators": True,
                       "mg_levels_patch_pc_patch_construct_type": "star",
                       "mg_levels_patch_pc_patch_construct_dim": 1,
                       "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
                       "mg_levels_patch_sub_ksp_type": "preonly",
                       "mg_levels_patch_sub_pc_type": "lu",
                       "mg_coarse_pc_type": "python",
                       "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                       "mg_coarse_assembled_pc_type": "lu",
                       "mg_coarse_assembled_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER}

    elif problem_type == "mixed":
        base = UnitSquareMesh(5, 5, distribution_parameters=distribution_parameters, quadrilateral=True)
        mh = MeshHierarchy(base, 1, distribution_parameters=distribution_parameters)
        mesh = mh[-1]
        V1 = VectorFunctionSpace(mesh, "CG", 2)
        V2 = FunctionSpace(mesh, "CG", 1)
        V = MixedFunctionSpace([V1, V2])

        u = Function(V)
        (z, p) = split(u)
        (v, q) = split(TestFunction(V))

        a = inner(grad(z), grad(v))*dx + inner(p, q)*dx

        bcs = DirichletBC(V.sub(0), Constant((1., 0.)), "on_boundary")
        nsp = MixedVectorSpaceBasis(
            V, [V.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)]
        )

        star_params = {"mat_type": "aij",
                       "snes_type": "ksponly",
                       "ksp_type": "richardson",
                       "pc_type": "mg",
                       "pc_mg_type": "multiplicative",
                       "pc_mg_cycle_type": "v",
                       "mg_levels_ksp_type": "chebyshev",
                       "mg_levels_ksp_max_it": 2,
                       "mg_levels_ksp_convergence_test": "skip",
                       "mg_levels_pc_type": "python",
                       "mg_levels_pc_python_type": "firedrake.ASMStarPC",
                       "mg_levels_pc_star_construct_dim": 0,
                       "mg_coarse_pc_type": "lu"}

        comp_params = {"mat_type": "aij",
                       "snes_type": "ksponly",
                       "ksp_type": "richardson",
                       "pc_type": "mg",
                       "pc_mg_type": "multiplicative",
                       "pc_mg_cycle_type": "v",
                       "mg_levels_ksp_type": "chebyshev",
                       "mg_levels_ksp_max_it": 2,
                       "mg_levels_ksp_convergence_test": "skip",
                       "mg_levels_pc_type": "python",
                       "mg_levels_pc_python_type": "firedrake.PatchPC",
                       "mg_levels_patch_pc_patch_save_operators": True,
                       "mg_levels_patch_pc_patch_construct_type": "star",
                       "mg_levels_patch_pc_patch_construct_dim": 0,
                       "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
                       "mg_levels_patch_sub_ksp_type": "preonly",
                       "mg_levels_patch_sub_pc_type": "lu",
                       "mg_coarse_pc_type": "python",
                       "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                       "mg_coarse_assembled_pc_type": "lu"}

    star_params["mg_levels_pc_star_backend"] = backend
    star_params["mg_levels_pc_star_mat_ordering_type"] = "rcm"
    nvproblem = NonlinearVariationalProblem(a, u, bcs=bcs)
    star_solver = NonlinearVariationalSolver(nvproblem, solver_parameters=star_params, nullspace=nsp)
    filter_warnings(star_solver.solve)
    star_its = star_solver.snes.getLinearSolveIterations()

    u.assign(0)
    comp_solver = NonlinearVariationalSolver(nvproblem, solver_parameters=comp_params, nullspace=nsp)
    filter_warnings(comp_solver.solve)
    comp_its = comp_solver.snes.getLinearSolveIterations()

    assert star_its == comp_its


def test_vanka_equivalence(problem_type):
    distribution_parameters = {"partition": True,
                               "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

    if problem_type == "scalar":
        base = UnitSquareMesh(10, 10, distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base, 2, distribution_parameters=distribution_parameters)
        mesh = mh[-1]
        V = FunctionSpace(mesh, "CG", 1)

        u = Function(V)
        v = TestFunction(V)

        a = inner(grad(u), grad(v))*dx - inner(Constant(1), v)*dx
        bcs = DirichletBC(V, 0, "on_boundary")
        nsp = None

        vanka_params = {"mat_type": "aij",
                        "snes_type": "ksponly",
                        "ksp_type": "richardson",
                        "pc_type": "mg",
                        "pc_mg_type": "multiplicative",
                        "pc_mg_cycle_type": "v",
                        "mg_levels_ksp_type": "richardson",
                        "mg_levels_ksp_richardson_scale": 1/10,
                        "mg_levels_ksp_max_it": 1,
                        "mg_levels_pc_type": "python",
                        "mg_levels_pc_python_type": "firedrake.ASMVankaPC",
                        "mg_levels_pc_vanka_construct_codim": 0,
                        "mg_coarse_pc_type": "lu",
                        "mg_coarse_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER}

        comp_params = {"mat_type": "aij",
                       "snes_type": "ksponly",
                       "ksp_type": "richardson",
                       "pc_type": "mg",
                       "pc_mg_type": "multiplicative",
                       "pc_mg_cycle_type": "v",
                       "mg_levels_ksp_type": "richardson",
                       "mg_levels_ksp_richardson_scale": 1/10,
                       "mg_levels_ksp_max_it": 1,
                       "mg_levels_pc_type": "python",
                       "mg_levels_pc_python_type": "firedrake.PatchPC",
                       "mg_levels_patch_pc_patch_save_operators": True,
                       "mg_levels_patch_pc_patch_construct_type": "vanka",
                       "mg_levels_patch_pc_patch_construct_codim": 0,
                       "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
                       "mg_levels_patch_sub_ksp_type": "preonly",
                       "mg_levels_patch_sub_pc_type": "lu",
                       "mg_coarse_pc_type": "python",
                       "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                       "mg_coarse_assembled_pc_type": "lu",
                       "mg_coarse_assembled_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER}

    elif problem_type == "vector":
        base = UnitSquareMesh(2, 2, distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base, 1, distribution_parameters=distribution_parameters)
        mesh = mh[-1]
        V = FunctionSpace(mesh, "RT", 1)

        u = Function(V)
        v = TestFunction(V)

        (x, y) = SpatialCoordinate(mesh)
        f = as_vector([x*(1-x), y*(1-y)])
        a = inner(div(u), div(v))*dx + inner(u, v)*dx - inner(f, v)*dx
        bcs = DirichletBC(V, 0, "on_boundary")
        nsp = None

        vanka_params = {"mat_type": "aij",
                        "snes_type": "ksponly",
                        "ksp_type": "cg",
                        "pc_type": "mg",
                        "pc_mg_type": "full",
                        "mg_levels_ksp_type": "richardson",
                        "mg_levels_ksp_richardson_scale": 3/10,
                        "mg_levels_ksp_max_it": 1,
                        "mg_levels_pc_type": "python",
                        "mg_levels_pc_python_type": "firedrake.ASMVankaPC",
                        "mg_levels_pc_vanka_construct_codim": 0,
                        "mg_coarse_pc_type": "lu",
                        "mg_coarse_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER}

        comp_params = {"mat_type": "aij",
                       "snes_type": "ksponly",
                       "ksp_type": "cg",
                       "pc_type": "mg",
                       "pc_mg_type": "full",
                       "mg_levels_ksp_type": "richardson",
                       "mg_levels_ksp_richardson_scale": 3/10,
                       "mg_levels_ksp_max_it": 1,
                       "mg_levels_pc_type": "python",
                       "mg_levels_pc_python_type": "firedrake.PatchPC",
                       "mg_levels_patch_pc_patch_save_operators": True,
                       "mg_levels_patch_pc_patch_construct_type": "vanka",
                       "mg_levels_patch_pc_patch_construct_codim": 0,
                       "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
                       "mg_levels_patch_sub_ksp_type": "preonly",
                       "mg_levels_patch_sub_pc_type": "lu",
                       "mg_coarse_pc_type": "python",
                       "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                       "mg_coarse_assembled_pc_type": "lu",
                       "mg_coarse_assembled_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER}

    elif problem_type == "mixed":
        base = UnitSquareMesh(5, 5, distribution_parameters=distribution_parameters, quadrilateral=True)
        mh = MeshHierarchy(base, 1, distribution_parameters=distribution_parameters)
        mesh = mh[-1]
        V1 = VectorFunctionSpace(mesh, "CG", 2)
        V2 = FunctionSpace(mesh, "CG", 1)
        V = MixedFunctionSpace([V1, V2])

        u = Function(V)
        (z, p) = split(u)
        (v, q) = split(TestFunction(V))

        a = inner(grad(z), grad(v))*dx - inner(p, div(v))*dx - inner(div(z), q)*dx

        bcs = DirichletBC(V.sub(0), Constant((1., 0.)), "on_boundary")
        nsp = MixedVectorSpaceBasis(
            V, [V.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)]
        )

        vanka_params = {"mat_type": "aij",
                        "snes_type": "ksponly",
                        "ksp_type": "richardson",
                        "pc_type": "mg",
                        "pc_mg_type": "multiplicative",
                        "pc_mg_cycle_type": "v",
                        "mg_levels_ksp_type": "chebyshev",
                        "mg_levels_ksp_max_it": 2,
                        "mg_levels_ksp_convergence_test": "skip",
                        "mg_levels_pc_type": "python",
                        "mg_levels_pc_python_type": "firedrake.ASMVankaPC",
                        "mg_levels_pc_vanka_construct_dim": 0,
                        "mg_levels_pc_vanka_exclude_subspaces": "1",
                        "mg_levels_pc_vanka_sub_sub_pc_type": "cholesky",
                        "mg_coarse_pc_type": "cholesky",
                        "mg_coarse_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER}

        comp_params = {"mat_type": "aij",
                       "snes_type": "ksponly",
                       "ksp_type": "richardson",
                       "pc_type": "mg",
                       "pc_mg_type": "multiplicative",
                       "pc_mg_cycle_type": "v",
                       "mg_levels_ksp_type": "chebyshev",
                       "mg_levels_ksp_max_it": 2,
                       "mg_levels_ksp_convergence_test": "skip",
                       "mg_levels_pc_type": "python",
                       "mg_levels_pc_python_type": "firedrake.PatchPC",
                       "mg_levels_patch_pc_patch_save_operators": True,
                       "mg_levels_patch_pc_patch_construct_type": "vanka",
                       "mg_levels_patch_pc_patch_construct_dim": 0,
                       "mg_levels_patch_pc_patch_exclude_subspaces": "1",
                       "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
                       "mg_levels_patch_sub_ksp_type": "preonly",
                       "mg_levels_patch_sub_pc_type": "lu",
                       "mg_coarse_pc_type": "python",
                       "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                       "mg_coarse_assembled_pc_type": "lu",
                       "mg_coarse_assembled_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER}

    vanka_params["mg_levels_pc_vanka_mat_ordering_type"] = "rcm"
    nvproblem = NonlinearVariationalProblem(a, u, bcs=bcs)
    star_solver = NonlinearVariationalSolver(nvproblem, solver_parameters=vanka_params, nullspace=nsp)
    filter_warnings(star_solver.solve)
    star_its = star_solver.snes.getLinearSolveIterations()

    u.assign(0)
    comp_solver = NonlinearVariationalSolver(nvproblem, solver_parameters=comp_params, nullspace=nsp)
    filter_warnings(comp_solver.solve)
    comp_its = comp_solver.snes.getLinearSolveIterations()

    assert star_its == comp_its


@pytest.fixture(params=["interval", "square"])
def base(request):
    dp = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    if request.param == "interval":
        return UnitIntervalMesh(5, distribution_parameters=dp)
    elif request.param == "square":
        return UnitSquareMesh(5, 5, quadrilateral=True, distribution_parameters=dp)


@pytest.mark.parametrize("family,degree", [("Q", 3), ("NCF", 3)])
@pytest.mark.parametrize("periodic", (False, True), ids=("extruded", "extruded-periodic"))
def test_asm_extruded_star(base, periodic, family, degree):
    mesh = ExtrudedMesh(base, 5, periodic=periodic)
    if mesh.topological_dimension() == 2:
        family = family.replace("N", "RT")
    V = FunctionSpace(mesh, family, degree)
    space = V.ufl_element().sobolev_space
    d = {H1: grad, HCurl: curl, HDiv: div}[space]

    x = SpatialCoordinate(mesh)
    uexact = x[0]*(1-x[0])
    if V.value_shape != uexact.ufl_shape:
        uexact = grad(uexact)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(d(u), d(v)) * dx + inner(u, v)*dx
    L = a(v, uexact)

    patch_dim = 1 if family == "NCF" else 0
    params = {
        "ksp_type": "cg",
        "pc_type": "python",
        "pc_python_type": "firedrake.P1PC",
        "pmg_mg_coarse_ksp_type": "preonly",
        "pmg_mg_coarse_pc_type": "lu",
        "pmg_mg_coarse_pc_factor_mat_solver_type": "mumps",
        "pmg_mg_levels_ksp_type": "chebyshev",
        "pmg_mg_levels_pc_type": "python",
        "pmg_mg_levels_pc_python_type": "firedrake.ASMExtrudedStarPC",
        "pmg_mg_levels_pc_star_construct_dim": patch_dim,
    }

    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    expected = 7
    assert solver.snes.getLinearSolveIterations() <= expected
    assert errornorm(uexact, uh) < 1E-7
