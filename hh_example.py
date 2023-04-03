
from firedrake import *

# number of elements in each direction
n = 5

# create mesh
mesh = UnitSquareMesh(n, n, quadrilateral=False)

# create function space
V = FunctionSpace(mesh, "CG", 1)

# Define the trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define the right-hand side
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate(sin(x)*cos(y))

# define the bilinear and linear forms for the LHS and RHS of the equation
LHS = (inner(grad(u), grad(v)) + inner(u, v)) * dx
RHS = inner(f, v) * dx

# Solve the problem
u_soln = Function(V)
solve(LHS == RHS, u_soln)