
"""This demo program solves the Poisson eigenvalue problem

  - div grad u(x) = lambda u(x) (STRONG FORM)
  grad u(x) dot grad v(x) dx = lambda u(x)v(x) dx (WEAK FORM)

on the unit interval (0, pi)
"""
from firedrake import *

# number of cells over the interval
n = 200

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
eigenprob = LinearEigenproblem(a, bcs=bc) #bcs=bc)

# Create corresponding eigensolver, looking for 5 eigenvalues 
eigensolver = LinearEigensolver(eigenprob, 5)

# Solve the problem
ncov = eigensolver.solve()

vr, vi = eigensolver.eigenfunction(0)

'''TESTING THE EVALS'''
def evals():
    print('eigensolver solns')
    # returns them in smallest to largest
    for i in range(ncov-2): # we consider the n-2 internal eigenvalues, final 2 correspond to boundary ones
        eval = eigensolver.eigenvalue(i)
        print(1/eval)

    print('BOFFI solns')
    h = pi /n
    for k in range(1, ncov-1):
        ans = 6 / h**2
        ans *= (1-cos(k*h))/(2+cos(k*h))
        print(ans)


evals()
