
"""This demo program solves the Poisson eigenvalue problem

  - div grad u(x) = lambda u(x) (STRONG FORM)
  grad u(x) dot grad v(x) dx = lambda u(x)v(x) dx (WEAK FORM)

on the unit interval (0, pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi) TO CHANGE
"""
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

# Create eigenproblem with boundary conditions
eigenprob = LinearEigenproblem(a, bcs=bc)

# Create corresponding eigensolver, looking for 1 eigenvalue
eigensolver = LinearEigensolver(eigenprob, 2)

# Solve the problem
ncov = eigensolver.solve()

vr, vi = eigensolver.eigenfunction(0)


'''TESTING THE EVALS'''
print('my fns')
for i in range(ncov):
    print(eigensolver.eigenvalue(i))
# View eigenvalues and eigenvectors
# eval_1 = eigensolver.eigenvalue(0)
# vr, vi = eigensolver.eigenfunction(0)



print('numerical')
h = pi /5
ans = 6 / h**2
for k in range(ncov):
    ans = 6 / h**2
    ans *= (1-cos(k*h))/(2+cos(k*h))
    print(ans)