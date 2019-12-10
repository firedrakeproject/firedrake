from firedrake import *
from firedrake.mg.ufl_utils import coarsen as symbolic_coarsen
from functools import singledispatch


def test_coarsen_callback():
    mesh = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(mesh, 1)
    mesh = mh[-1]

    V = FunctionSpace(mesh, "CG", 3)

    u = TrialFunction(V)

    v = TestFunction(V)

    a = dot(grad(u), grad(v))*dx + u*v*dx

    L = Constant(1)*v*dx

    @singledispatch
    def coarsen(expr, self, coefficient_mapping=None):
        return symbolic_coarsen(expr, self, coefficient_mapping=coefficient_mapping)

    @coarsen.register(functionspaceimpl.FunctionSpace)
    @coarsen.register(functionspaceimpl.WithGeometry)
    def coarsen_fs(V, self, coefficient_mapping=None):
        mesh = self(V.ufl_domain(), self)
        return FunctionSpace(mesh, "CG", 1)

    uh = Function(V)
    lvp = LinearVariationalProblem(a, L, uh)
    lvs = LinearVariationalSolver(lvp, solver_parameters={"ksp_type": "cg",
                                                          "pc_type": "mg"})
    with dmhooks.ctx_coarsener(V, coarsen):
        lvs.solve()

    Ac, _ = lvs.snes.ksp.pc.getMGCoarseSolve().getOperators()

    assert Ac.getSize() == (25, 25)


def test_sphere_mg():
    R = 1.0

    base = IcosahedralSphereMesh(radius=R, refinement_level=0)
    nref = 5
    mh = MeshHierarchy(base, nref)
    for mesh in mh:
        x = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)

    mesh = mh[-1]

    x, y, z = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = (u*v + inner(u, v))*dx

    f1 = exp((x+y+z)/R)*x*y*z/R**3
    F = v*f1*dx

    mg_params = {"mat_type": "matfree",
                 "snes_type": "ksponly",
                 "ksp_type": "gmres",
                 "ksp_rtol": 1.0e-8,
                 "ksp_atol": 0.0,
                 "ksp_max_it": 1000,
                 "ksp_monitor": None,
                 "ksp_converged_reason": None,
                 "ksp_norm_type": "unpreconditioned",
                 "pc_type": "mg",
                 "mg_coarse_ksp_type": "preonly",
                 "mg_coarse_pc_type": "python",
                 "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                 "mg_coarse_assembled_pc_type": "lu",
                 "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
                 "mg_coarse_assembled_mat_mumps_icntl_14": 200,
                 "mg_levels_pc_type": "python",
                 "mg_levels_pc_python_type": "firedrake.AssembledPC",
                 "mg_levels_assembled_pc_type": "bjacobi",
                 "mg_levels_assembled_sub_pc_type": "ilu",
                 "mg_levels_ksp_type": "richardson",
                 "mg_levels_ksp_max_it": 1}

    w = Function(V)

    Prob = LinearVariationalProblem(a, F, w)
    Solver = LinearVariationalSolver(Prob,
                                     solver_parameters=mg_params)
    Solver.solve()
    assert(Solver.snes.ksp.getIterationNumber() < 5)
