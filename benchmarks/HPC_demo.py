from firedrake import *
from firedrake.petsc import PETSc
from time import time


parprint = PETSc.Sys.Print

Nx = 8
Nref = 2
degree = 2

# Create mesh and mesh hierarchy
mesh = UnitCubeMesh(Nx, Nx, Nx)
hierarchy = MeshHierarchy(mesh, Nref)
mesh = hierarchy[-1]

# Define the function space and print the DOFs
V = FunctionSpace(mesh, "CG", degree)
dofs = V.dim()
parprint('DOFs', dofs)

u = TrialFunction(V)
v = TestFunction(V)

bcs = DirichletBC(V, zero(), ("on_boundary",))

# Define the RHS and analytic solution
x, y, z = SpatialCoordinate(mesh)

a = Constant(1)
b = Constant(2)
exact = sin(pi*x)*tan(pi*x/4)*sin(a*pi*y)*sin(b*pi*z)
truth = Function(V).interpolate(exact)
f = -pi**2 / 2
f *= 2*cos(pi*x) - cos(pi*x/2) - 2*(a**2 + b**2)*sin(pi*x)*tan(pi*x/4)
f *= sin(a*pi*y)*sin(b*pi*z)

# Define the problem using the bilinear form `a` and linear functional `L`
a = dot(grad(u), grad(v))*dx
L = f*v*dx
u_h = Function(V)
problem = LinearVariationalProblem(a, L, u_h, bcs=bcs)


def run_solve(problem, parameters):
    # Create a solver and time how long the solve takes
    t = time()
    solver = LinearVariationalSolver(problem, solver_parameters=parameters)
    solver.solve()
    parprint("Runtime :", time() - t)


# Direct solve with LU
lu_mumps = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

# Conjugate Gradient (CG) with Algebraic Multigrid (AMG)
cg_amg = {
    "snes_view": None,
    "ksp_type": "cg",
    "pc_type": "gamg",
    "pc_mg_log": None
}

# CG with Geometric Multigrid (GMG) V-cycles
cg_gmg_v = {
    "snes_view": None,
    "ksp_type": "cg",
    "pc_type": "mg",
    "pc_mg_log": None
}

# CG + GMG F-cycles
fmg = {
    "snes_view": None,
    "ksp_type": "cg",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_pc_type": "jacobi",
    "mg_coarse_pc_type": "lu",
    "mg_coarse_pc_factor_mat_solver_type": "mumps"
}

# CG + GMG F-cycles and telescoping
telescope_factor = 1 # Set to number of nodes!
fmg_matfree_telescope = {
    "snes_view": None,
    "mat_type": "matfree",
    "ksp_type": "cg",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_pc_type": "jacobi",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled": {
        "mat_type": "aij",
        "pc_type": "telescope",
        "pc_telescope_reduction_factor": telescope_factor,
        "pc_telescope_subcomm_type": "contiguous",
        "telescope_pc_type": "lu",
        "telescope_pc_factor_mat_solver_type": "mumps"
    }
}

## Run the solve
u_h.assign(0)
run_solve(problem, fmg_matfree_telescope)
parprint("Error   :", errornorm(truth, u_h))
