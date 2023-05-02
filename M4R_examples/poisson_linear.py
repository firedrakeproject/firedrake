from firedrake import *

# number of elements in each direction
n = 5

# create mesh
mesh = IntervalMesh(n, 0, pi)

# create function space
V = FunctionSpace(mesh, "CG", 1)

# Define the trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define the variational form
a = (inner(grad(u), grad(v))) * dx

# Apply the homogeneous Dirichlet boundary conditions
bc = None#DirichletBC(V, 0.0, "on_boundary")

# Create linear variational problem
eigenprob = LinearVariationalProblem(a, bcs=bc)

# Create corresponding eigensolver, looking for 2 eigenvalues
eigensolver = LinearEigensolver(eigenprob, 2)

# Solve the problem
eigensolver.solve()

# Store eigenvalues and eigenvectors
eval_1 = eigensolver.eigenvalue(0)
vr, vi = eigensolver.eigenfunction(0)

