import pytest
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
def CG1(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture(scope='module', params=[0, 1, 2])
def solver_params(request):
    if request.param == 0:
        return {
               "mat_type": "matfree",  # noqa: E126
               "snes_type": "fas",
               "snes_fas_cycles": 1,
               "snes_fas_type": "full",
               "snes_fas_galerkin": False,
               "snes_fas_smoothup": 1,
               "snes_fas_smoothdown": 1,
               "snes_monitor": None,
               "snes_max_it": 20,
               "fas_levels_snes_type": "python",
               "fas_levels_snes_python_type": "firedrake.PatchSNES",
               "fas_levels_snes_max_it": 1,
               "fas_levels_snes_convergence_test": "skip",
               "fas_levels_snes_converged_reason": None,
               "fas_levels_snes_monitor": None,
               "fas_levels_snes_linesearch_type": "basic",
               "fas_levels_snes_linesearch_damping": 4/5,
               "fas_levels_patch_snes_patch_partition_of_unity": False,
               "fas_levels_patch_snes_patch_construct_type": "star",
               "fas_levels_patch_snes_patch_construct_dim": 0,
               "fas_levels_patch_snes_patch_sub_mat_type": "seqdense",
               "fas_levels_patch_snes_patch_local_type": "additive",
               "fas_levels_patch_snes_patch_symmetrise_sweep": False,
               "fas_levels_patch_sub_snes_type": "newtonls",
               "fas_levels_patch_sub_snes_converged_reason": None,
               "fas_levels_patch_sub_snes_linesearch_type": "basic",
               "fas_levels_patch_sub_ksp_type": "preonly",
               "fas_levels_patch_sub_pc_type": "lu",
               "fas_coarse_snes_type": "newtonls",
               "fas_coarse_snes_monitor": None,
               "fas_coarse_snes_converged_reason": None,
               "fas_coarse_snes_max_it": 100,
               "fas_coarse_snes_atol": 1.0e-14,
               "fas_coarse_snes_rtol": 1.0e-14,
               "fas_coarse_snes_linesearch_type": "l2",
               "fas_coarse_ksp_type": "preonly",
               "fas_coarse_ksp_max_it": 1,
               "fas_coarse_pc_type": "python",
               "fas_coarse_pc_python_type": "firedrake.AssembledPC",
               "fas_coarse_assembled_mat_type": "aij",
               "fas_coarse_assembled_pc_type": "lu",
               "fas_coarse_assembled_pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS,
               "snes_view": None
        }
    elif request.param == 1:
        return {
               "mat_type": "matfree",  # noqa: E126
               "snes_type": "fas",
               "snes_fas_cycles": 1,
               "snes_fas_type": "full",
               "snes_fas_galerkin": False,
               "snes_fas_smoothup": 1,
               "snes_fas_smoothdown": 1,
               "snes_monitor": None,
               "snes_max_it": 20,
               "fas_levels_snes_type": "python",
               "fas_levels_snes_python_type": "firedrake.PatchSNES",
               "fas_levels_snes_max_it": 1,
               "fas_levels_snes_convergence_test": "skip",
               "fas_levels_snes_converged_reason": None,
               "fas_levels_snes_monitor": None,
               "fas_levels_snes_linesearch_type": "basic",
               "fas_levels_snes_linesearch_damping": 4/5,
               "fas_levels_patch_snes_patch_partition_of_unity": False,
               "fas_levels_patch_snes_patch_construct_type": "vanka",
               "fas_levels_patch_snes_patch_construct_dim": 0,
               "fas_levels_patch_snes_patch_vanka_dim": 0,
               "fas_levels_patch_snes_patch_sub_mat_type": "seqdense",
               "fas_levels_patch_snes_patch_local_type": "additive",
               "fas_levels_patch_snes_patch_symmetrise_sweep": False,
               "fas_levels_patch_sub_snes_type": "newtonls",
               "fas_levels_patch_sub_snes_converged_reason": None,
               "fas_levels_patch_sub_snes_linesearch_type": "basic",
               "fas_levels_patch_sub_ksp_type": "preonly",
               "fas_levels_patch_sub_pc_type": "lu",
               "fas_coarse_snes_type": "newtonls",
               "fas_coarse_snes_monitor": None,
               "fas_coarse_snes_converged_reason": None,
               "fas_coarse_snes_max_it": 100,
               "fas_coarse_snes_atol": 1.0e-14,
               "fas_coarse_snes_rtol": 1.0e-14,
               "fas_coarse_snes_linesearch_type": "l2",
               "fas_coarse_ksp_type": "preonly",
               "fas_coarse_ksp_max_it": 1,
               "fas_coarse_pc_type": "python",
               "fas_coarse_pc_python_type": "firedrake.AssembledPC",
               "fas_coarse_assembled_mat_type": "aij",
               "fas_coarse_assembled_pc_type": "lu",
               "fas_coarse_assembled_pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS,
               "snes_view": None
        }
    else:
        return {
               "mat_type": "matfree",  # noqa: E126
               "snes_type": "fas",
               "snes_fas_cycles": 1,
               "snes_fas_type": "full",
               "snes_fas_galerkin": False,
               "snes_fas_smoothup": 1,
               "snes_fas_smoothdown": 1,
               "snes_fas_full_downsweep": False,
               "snes_monitor": None,
               "snes_max_it": 20,
               "fas_levels_snes_type": "python",
               "fas_levels_snes_python_type": "firedrake.PatchSNES",
               "fas_levels_snes_max_it": 1,
               "fas_levels_snes_convergence_test": "skip",
               "fas_levels_snes_converged_reason": None,
               "fas_levels_snes_monitor": None,
               "fas_levels_snes_linesearch_type": "basic",
               "fas_levels_snes_linesearch_damping": 1.0,
               "fas_levels_patch_snes_patch_construct_type": "pardecomp",
               "fas_levels_patch_snes_patch_partition_of_unity": True,
               "fas_levels_patch_snes_patch_pardecomp_overlap": 1,
               "fas_levels_patch_snes_patch_sub_mat_type": "seqaij",
               "fas_levels_patch_snes_patch_local_type": "additive",
               "fas_levels_patch_snes_patch_symmetrise_sweep": False,
               "fas_levels_patch_sub_snes_type": "newtonls",
               "fas_levels_patch_sub_snes_monitor": None,
               "fas_levels_patch_sub_snes_converged_reason": None,
               "fas_levels_patch_sub_snes_linesearch_type": "basic",
               "fas_levels_patch_sub_ksp_type": "preonly",
               "fas_levels_patch_sub_pc_type": "lu",
               "fas_coarse_snes_type": "newtonls",
               "fas_coarse_snes_monitor": None,
               "fas_coarse_snes_converged_reason": None,
               "fas_coarse_snes_max_it": 100,
               "fas_coarse_snes_atol": 1.0e-14,
               "fas_coarse_snes_rtol": 1.0e-14,
               "fas_coarse_snes_linesearch_type": "l2",
               "fas_coarse_ksp_type": "preonly",
               "fas_coarse_ksp_max_it": 1,
               "fas_coarse_pc_type": "python",
               "fas_coarse_pc_python_type": "firedrake.AssembledPC",
               "fas_coarse_assembled_mat_type": "aij",
               "fas_coarse_assembled_pc_type": "lu",
               "fas_coarse_assembled_pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS,
        }


@pytest.mark.parallel
def test_snespatch(mesh, CG1, solver_params):
    u = Function(CG1)
    v = TestFunction(CG1)

    f = Constant(1, domain=mesh)
    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx + inner(u**3 - u, v)*dx

    bcs = DirichletBC(CG1, 0, "on_boundary")

    nvproblem = NonlinearVariationalProblem(F, u, bcs=bcs)
    solver = NonlinearVariationalSolver(nvproblem, solver_parameters=solver_params)
    solver.solve()

    assert solver.snes.reason > 0
