import pytest
import numpy
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER_PARAMETERS


@pytest.fixture(scope='module')
def mesh():
    N = 10
    nref = 1
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    base = UnitSquareMesh(N, N, distribution_parameters=distribution_parameters)
    mh = MeshHierarchy(base, nref, distribution_parameters=distribution_parameters)
    return mh[-1]


@pytest.fixture(scope='module')
def V(mesh):
    return VectorFunctionSpace(mesh, "CG", 4)


def test_patch_precompute_element_tensors(mesh, V):
    u = Function(V)
    v = TestFunction(V)

    gamma = Constant(1)
    f = Constant((1, 1))
    F = inner(grad(u), grad(v))*dx + gamma*inner(div(u), div(v))*dx - inner(f, v)*dx + avg(inner(u, v))*dS

    bcs = DirichletBC(V, 0, "on_boundary")

    sp = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-8,
        "ksp_atol": 0.0,
        "ksp_max_it": 1000,
        "ksp_converged_reason": None,
        "ksp_norm_type": "unpreconditioned",
        "pc_type": "mg",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
        "mg_coarse_assembled_pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS,
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_max_it": 1,
        "mg_levels_ksp_richardson_scale": 1/3,
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "firedrake.PatchPC",
        "mg_levels_patch_pc_patch_save_operators": True,
        "mg_levels_patch_pc_patch_partition_of_unity": False,
        "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
        "mg_levels_patch_pc_patch_construct_type": "star",
        "mg_levels_patch_pc_patch_multiplicative": False,
        "mg_levels_patch_pc_patch_symmetrise_sweep": False,
        "mg_levels_patch_pc_patch_construct_dim": 0,
        "mg_levels_patch_sub_ksp_type": "preonly",
        "mg_levels_patch_sub_pc_type": "lu",
    }

    nvproblem = NonlinearVariationalProblem(F, u, bcs=bcs)
    solver = NonlinearVariationalSolver(nvproblem, solver_parameters=sp)

    history_noprecompute = []
    for gamma_ in [1e3, 1e4, 1e5]:
        gamma.assign(gamma_)

        solver.snes.ksp.setConvergenceHistory()
        u.assign(0)
        solver.solve()
        history_noprecompute.append(solver.snes.ksp.getConvergenceHistory())

    precompute = sp.copy()
    precompute["mg_levels_patch_pc_patch_precompute_element_tensors"] = True
    solver = NonlinearVariationalSolver(nvproblem, solver_parameters=precompute)

    history_precompute = []
    for gamma_ in [1e3, 1e4, 1e5]:
        gamma.assign(gamma_)

        solver.snes.ksp.setConvergenceHistory()
        u.assign(0)
        solver.solve()
        history_precompute.append(solver.snes.ksp.getConvergenceHistory())

    for (yes, no) in zip(history_precompute, history_noprecompute):
        assert numpy.allclose(yes, no)
