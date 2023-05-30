
"""This demo program solves the Poisson eigenvalue problem

  - div grad u(x) = lambda u(x) (STRONG FORM)
  grad u(x) dot grad v(x) dx = lambda u(x)v(x) dx (WEAK FORM)

on the unit interval (0, pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi) TO CHANGE
"""
from firedrake import *

# number of elements in each direction
n = 6

# create mesh
mesh = IntervalMesh(n, 0, 1)

# create function space
V = FunctionSpace(mesh, "CG", 1)

# Define the trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define the variational form
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx

# Apply the homogeneous Dirichlet boundary conditions
#bc = DirichletBC(V, 0.0, "on_boundary")

# Create eigenproblem with boundary conditions
eigenprob = LinearEigenproblem(a)#, bcs=bc)

# Create corresponding eigensolver, looking for 1 eigenvalue
eigensolver = LinearEigensolver(eigenprob, 4)

# Solve the problem
ncov = eigensolver.solve()

vr, vi = eigensolver.eigenfunction(0)


'''TESTING THE EVALS'''
print('eigensolver solns')
for i in range(ncov):
    bla = eigensolver.eigenvalue(i)
    print(1/bla)

print('analytic solns')
h = 1 /n
for k in range(1, ncov + 1):
    #lam = (1 / h**2) * (2 - 2 * cos(k * pi * h)) - 1
    #print(lam)
    print(k*pi)