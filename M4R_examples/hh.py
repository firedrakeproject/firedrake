from firedrake import *

# Number of elements in each direction
n = 5

# Create mesh
mesh = UnitSquareMesh(n, n, quadrilateral=False)

# Create function space
V = FunctionSpace(mesh, "CG", 1)

# Define the trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
