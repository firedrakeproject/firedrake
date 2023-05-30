
"""This demo program solves the Poisson eigenvalue problem

  - div grad u(x) = lambda u(x) (STRONG FORM)
  grad u(x) dot grad v(x) dx = lambda u(x)v(x) dx (WEAK FORM)

on the 2d domain  (0, pi) x (0, pi) 

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi) TO CHANGE
"""
from firedrake import *

# number of elements in each direction
n = 4 # 10 x 10 elements

# create mesh
mesh = UnitSquareMesh(n, n, quadrilateral=False) 

# create function space
V = FunctionSpace(mesh, "CG", 1) # piecewise linear finite elements on triangles

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
eigensolver = LinearEigensolver(eigenprob, 4)

# Solve the problem
ncov = eigensolver.solve()

vr, vi = eigensolver.eigenfunction(0)


'''TESTING THE EVALS'''
print('eigensolver solns')
for i in range(ncov):
    bla = eigensolver.eigenvalue(i)
    print(1/bla)
    # View eigenvalues and eigenvectors
# eval_1 = eigensolver.eigenvalue(0)
# vr, vi = eigensolver.eigenfunction(0)



print('BOFFI solns')
for m in range(ncov):
    for n in range(ncov):
        print(m**2 + n**2)
# h = pi /n
# for n in range(1, ncov + 1):
#     ans = 6 / h**2
#     ans *= (1-cos(k*h))/(2+cos(k*h))
#     print(ans)