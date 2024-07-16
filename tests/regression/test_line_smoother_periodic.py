from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER_PARAMETERS
from firedrake.utils import RealType
import pytest


# Useful for making a periodic hierarchy
def periodise(m):
    element = BrokenElement(FiniteElement("CG", cell=m.ufl_cell(), degree=1))
    coord_fs = VectorFunctionSpace(m, element, dim=2)
    old_coordinates = m.coordinates
    new_coordinates = Function(coord_fs)
    domain = "{[i, j]: 0 <= i < old_coords.dofs and 0 <= j < new_coords.dofs}"
    instructions = f"""
    <{RealType}> Y = 0
    <float64> pi = 3.141592653589793
    for i
        Y = Y + real(old_coords[i, 1])
    end
    for j
        new_coords[j, 0] = atan2(real(old_coords[j, 1]), real(old_coords[j, 0])) / (pi*2)
        new_coords[j, 0] = new_coords[j, 0] + 1 if real(new_coords[j, 0]) < 0 else new_coords[j, 0]
        new_coords[j, 0] = 1 if (real(new_coords[j, 0]) == 0 and Y < 0) else new_coords[j, 0]
        new_coords[j, 0] = new_coords[j, 0] * Lx[0]
        new_coords[j, 1] = old_coords[j, 2] * Ly[0]
    end
    """
    cLx = Constant(1)
    cLy = Constant(1)
    par_loop((domain, instructions), dx,
             {"new_coords": (new_coordinates, WRITE),
              "old_coords": (old_coordinates, READ),
              "Lx": (cLx, READ),
              "Ly": (cLy, READ)})
    return Mesh(new_coordinates)


@pytest.mark.skipcomplex
def test_line_smoother_periodic():
    N = 3
    H = 0.1
    nsmooth = 3
    nref = 1

    # Making a periodic hierarchy:
    # first make a hierarchy of the cylinder meshes,
    # then periodise each one
    baseMesh = CylinderMesh(N, N, 1.0, H)

    mh = MeshHierarchy(baseMesh, nref)
    meshes = tuple(periodise(m) for m in mh)
    mh = HierarchyBase(meshes, mh.coarse_to_fine_cells, mh.fine_to_coarse_cells)
    mesh = mh[-1]

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(0), [1, 2])

    F = inner(grad(u), grad(v))*dx - inner(Constant(1), v)*dx

    base = {"snes_type": "ksponly",
            "ksp_type": "fgmres",
            "ksp_monitor": None,
            "ksp_max_it": 100,
            "ksp_rtol": 1.0e-5}

    mg_levels = {"ksp_max_it": nsmooth,
                 "ksp_monitor": None,
                 "ksp_norm_type": "unpreconditioned",
                 "pc_type": "python",
                 "pc_python_type": "firedrake.PatchPC",
                 "patch_pc_patch_save_operators": True,
                 "patch_pc_patch_local_type": "additive",
                 "patch_pc_patch_construct_type": "python",
                 "patch_pc_patch_construct_python_type": "firedrake.PlaneSmoother",
                 "patch_pc_patch_construct_ps_sweeps": "0-%d" % (N+1),
                 "patch_sub_ksp_type": "preonly",
                 "patch_sub_pc_type": "lu"}

    params = {
        "ksp_type": "richardson",
        "ksp_richardson_self_scale": False,
        "ksp_norm_type": "unpreconditioned",
        "pc_type": "mg",
        "pc_mg_type": "full",
        "pc_mg_log": None,
        "mg_levels": mg_levels,
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled": {
            "mat_type": "aij",
            "pc_type": "telescope",
            "pc_telescope_subcomm_type": "contiguous",
            "telescope_pc_type": "lu",
            "telescope_pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS
        }
    }

    params = {**base, **params}

    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    assert solver.snes.ksp.its <= 5
