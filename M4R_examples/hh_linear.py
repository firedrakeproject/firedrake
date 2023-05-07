from firedrake import *
# Number of elements in each direction
n = 5
# Create mesh
mesh = UnitSquareMesh(n, n, quadrilateral=False)
# Create function space
V = FunctionSpace(mesh, "CG", 1)
# Define the trial and test functions
u = TrialFuncton(V)
v = TestFunction(V)
# Define the right-hand side
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate(sin(x)*cos(y))
# Define the bilinear and linear forms for the LHS and RHS of the equation
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx  # LHS
L = inner(f, v) * dx  # RHS
params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}# can get rid of params
bcs = None  # CHANGE THIS TO APPLY BOUNDARY CONDITIONS
u_soln = Function(V)  # function to hold the solution
# Create the Linear Variational Problem
problem = LinearVariationalProblem(a, L, u_soln, bcs=bcs)
# Create the Linear Variational Solver
solver = LinearVariationalSolver(problem, solver_parameters=params)  
# Solve the problem
solver.solve()
