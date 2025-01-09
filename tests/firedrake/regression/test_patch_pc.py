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
               for Q in V.subfunctions]
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
