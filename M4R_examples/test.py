from firedrake import *

# number of elements in each direction
n = 10

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
bc = DirichletBC(V, 0.0, "on_boundary")

# Create eigenproblem
eigenprob = LinearEigenproblem(a)

# Create corresponding eigensolver, looking for 1 eigenvalue
eigensolver = LinearEigensolver(eigenprob, 2)

# Solve the problem
eigensolver.solve()

# View eigenvalues and eigenvectors
eval_1 = eigensolver.eigenvalue(0)
