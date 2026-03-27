import pytest
import numpy
from firedrake import *


@pytest.fixture(params=[1, 2, 3],
                ids=["Interval", "Rectangle", "Box"])
def mesh(request):
    distribution = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    if request.param == 1:
        return IntervalMesh(3, 5, distribution_parameters=distribution)
    if request.param == 2:
        return RectangleMesh(10, 20, 2, 3, distribution_parameters=distribution)
    if request.param == 3:
        return BoxMesh(5, 3, 5, 1, 2, 3, distribution_parameters=distribution)


@pytest.fixture(params=["scalar", "vector", "tensor", "mixed"])
def problem_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def multiplicative(request):
    return request.param


def test_jacobi_sor_equivalence(mesh, problem_type, multiplicative):
    if problem_type == "scalar":
        V = FunctionSpace(mesh, "CG", 1)
    elif problem_type == "vector":
        V = VectorFunctionSpace(mesh, "CG", 1)
    elif problem_type == "tensor":
        V = TensorFunctionSpace(mesh, "CG", 1)
    elif problem_type == "mixed":
        P = FunctionSpace(mesh, "CG", 1)
        Q = VectorFunctionSpace(mesh, "CG", 1)
        R = TensorFunctionSpace(mesh, "CG", 1)
        V = P*Q*R

    shape = V.value_shape
    rhs = numpy.full(shape, 1, dtype=float)

    u = TrialFunction(V)
    v = TestFunction(V)

    if problem_type == "mixed":
        # We also test patch pc with kernel argument compression.
        i = 1  # only active index
        f = Function(V)
        fval = numpy.full(V.sub(i).value_shape, 1.0, dtype=float)
        f.sub(i).interpolate(Constant(fval))
        a = (inner(f[i], f[i]) * inner(grad(u), grad(v)))*dx
        L = inner(Constant(rhs), v)*dx
        bcs = [DirichletBC(Q, 0, "on_boundary")
               for Q in V.subspaces]
    else:
        a = inner(grad(u), grad(v))*dx
        L = inner(Constant(rhs), v)*dx
        bcs = DirichletBC(V, 0, "on_boundary")

    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    jacobi = LinearVariationalSolver(problem,
                                     solver_parameters={"ksp_type": "cg",
                                                        "pc_type": "sor" if multiplicative else "jacobi",
                                                        "ksp_monitor": None,
                                                        "mat_type": "aij"})

    jacobi.snes.ksp.setConvergenceHistory()

    jacobi.solve()

    jacobi_history = jacobi.snes.ksp.getConvergenceHistory()

    patch = LinearVariationalSolver(problem,
                                    options_prefix="",
                                    solver_parameters={"mat_type": "matfree",
                                                       "ksp_type": "cg",
                                                       "pc_type": "python",
                                                       "pc_python_type": "firedrake.PatchPC",
                                                       "patch_pc_patch_construct_type": "star",
                                                       "patch_pc_patch_save_operators": True,
                                                       "patch_pc_patch_sub_mat_type": "aij",
                                                       "patch_pc_patch_local_type": "multiplicative" if multiplicative else "additive",
                                                       "patch_pc_patch_symmetrise_sweep": multiplicative,
                                                       "patch_sub_ksp_type": "preonly",
                                                       "patch_sub_pc_type": "lu",
                                                       "ksp_monitor": None})

    patch.snes.ksp.setConvergenceHistory()

    uh.assign(0)
    patch.solve()

    patch_history = patch.snes.ksp.getConvergenceHistory()

    assert numpy.allclose(jacobi_history, patch_history)


def _patch_pc_exterior_facets_problem(a, L):
    """Helper: solve with ASMStarPC and PatchPC, return iteration counts."""
    V = a.arguments()[0].function_space()

    u_star = Function(V)
    problem = LinearVariationalProblem(a, L, u_star)
    star_solver = LinearVariationalSolver(
        problem,
        solver_parameters={
            "mat_type": "aij",
            "ksp_type": "gmres",
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star_construct_dim": 0,
            "ksp_rtol": 1e-12,
        },
    )
    star_solver.snes.ksp.setConvergenceHistory()
    star_solver.solve()
    star_its = len(star_solver.snes.ksp.getConvergenceHistory())

    u_patch = Function(V)
    problem_patch = LinearVariationalProblem(a, L, u_patch)
    patch_solver = LinearVariationalSolver(
        problem_patch,
        options_prefix="",
        solver_parameters={
            "mat_type": "matfree",
            "ksp_type": "gmres",
            "pc_type": "python",
            "pc_python_type": "firedrake.PatchPC",
            "patch_pc_patch_construct_type": "star",
            "patch_pc_patch_construct_dim": 0,
            "patch_pc_patch_save_operators": True,
            "patch_sub_ksp_type": "preonly",
            "patch_sub_pc_type": "lu",
            "ksp_rtol": 1e-12,
        },
    )
    patch_solver.snes.ksp.setConvergenceHistory()
    patch_solver.solve()
    patch_its = len(patch_solver.snes.ksp.getConvergenceHistory())

    return star_its, patch_its


@pytest.mark.parallel([1, 3])
def test_patch_pc_exterior_facets_dx_ds():
    """Test that PatchPC correctly handles exterior facet integrals (ds)
    in both serial and parallel, by asserting it takes the same number
    of iterations as ASMStarPC."""
    distribution = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    mesh = UnitSquareMesh(4, 4, distribution_parameters=distribution)
    V = FunctionSpace(mesh, "DG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * dx + inner(u, v) * ds
    L = inner(Constant(1.0), v) * dx
    star_its, patch_its = _patch_pc_exterior_facets_problem(a, L)
    assert star_its == patch_its


def test_patch_pc_exterior_facets_dx_dS_ds():
    """Test that PatchPC correctly handles exterior (ds) and interior (dS)
    facet integrals together, by asserting it takes the same number of
    iterations as ASMStarPC."""
    distribution = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    mesh = UnitSquareMesh(4, 4, distribution_parameters=distribution)
    V = FunctionSpace(mesh, "DG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * dx + inner(avg(u), avg(v)) * dS + inner(u, v) * ds
    L = inner(Constant(1.0), v) * dx
    star_its, patch_its = _patch_pc_exterior_facets_problem(a, L)
    assert star_its == patch_its
