import pytest
from firedrake import *


@pytest.fixture(scope="module")
def hierarchy():
    N = 10
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    base = RectangleMesh(N, N, 2, 2, distribution_parameters=distribution_parameters)

    mh = MeshHierarchy(base, 3, distribution_parameters=distribution_parameters)
    for m in mh:
        m.coordinates.dat.data[:, 0] -= 1
        m.coordinates.dat.data[:, 1] -= 1
    return mh


@pytest.fixture
def mesh(hierarchy):
    return hierarchy[-1]


@pytest.fixture(params=[1, 2])
def degree(request):
    return request.param


@pytest.fixture(params=["RT", "N1curl", "CG"])
def space(request):
    return request.param


@pytest.fixture
def V(mesh, degree, space):
    return FunctionSpace(mesh, space, degree, variant="integral")


@pytest.fixture(params=["Default", "Exact", "Averaging"])
def use_averaging(request):
    return request.param


@pytest.fixture
def solver_parameters(use_averaging, V):
    element_name = V.ufl_element()._short_name
    solver_parameters = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        # When using mass solves in the prolongation, the V-cycle is
        # no longer a linear operator (because the prolongation uses
        # CG which is a nonlinear operator).
        "ksp_type": "cg" if use_averaging else "fcg",
        "ksp_max_it": 20,
        "ksp_rtol": 1e-9,
        "ksp_monitor_true_residual": None,
        "pc_type": "mg",
        "mg_levels": {
            "ksp_type": "richardson",
            "ksp_norm_type": "unpreconditioned",
            "ksp_richardson_scale": 0.5,
            "pc_type": "python",
            "pc_python_type": "firedrake.PatchPC",
            "patch_pc_patch_save_operators": True,
            "patch_pc_patch_partition_of_unity": False,
            "patch_pc_patch_construct_type": "star",
            "patch_pc_patch_construct_dim": 0,
            "patch_pc_patch_sub_mat_type": "seqdense",
            "patch_sub_ksp_type": "preonly",
            "patch_sub_pc_type": "lu",
        },
        "mg_coarse_pc_type": "lu",
        element_name: {
            "prolongation_mass_ksp_type": "cg",
            "prolongation_mass_ksp_max_it": 10,
            "prolongation_mass_pc_type": "bjacobi",
            "prolongation_mass_sub_pc_type": "ilu",
        }
    }
    return solver_parameters


@pytest.fixture
def solver(V, space, solver_parameters):
    u = Function(V)
    v = TestFunction(V)
    mesh = V.mesh()
    (x, y) = SpatialCoordinate(mesh)
    f = as_vector([2*y*(1-x**2),
                   -2*x*(1-y**2)])
    a = Constant(1)
    b = Constant(100)
    if space == "RT":
        F = a*inner(u, v)*dx + b*inner(div(u), div(v))*dx - inner(f, v)*dx
    elif space == "N1curl":
        F = a*inner(u, v)*dx + b*inner(curl(u), curl(v))*dx - inner(f, v)*dx
    elif space == "CG":
        F = a*inner(u, v)*dx + b*inner(grad(u), grad(v))*dx - inner(1, v)*dx
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                        options_prefix="")
    return solver


@pytest.mark.skipcomplexnoslate
def test_riesz(V, solver, use_averaging):
    if use_averaging == "Default":
        transfer = None
    elif use_averaging == "Exact":
        transfer = TransferManager(use_averaging=False)
    else:
        transfer = TransferManager(use_averaging=True)
    solver.set_transfer_manager(transfer)
    solver.solve()

    assert solver.snes.ksp.getIterationNumber() < 15


@pytest.mark.parallel(nprocs=3)
@pytest.mark.skipcomplexnoslate
def test_riesz_parallel(V, solver, use_averaging):
    test_riesz(V, solver, use_averaging)
