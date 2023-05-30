
"""This demo program solves the Poisson eigenvalue problem

  - div grad u(x) = lambda u(x) (STRONG FORM)
  grad u(x) dot grad v(x) dx = lambda u(x)v(x) dx (WEAK FORM)

on the unit interval (0, pi)
"""
from firedrake import *
import numpy as np  
import matplotlib.pyplot as plt

# number of cells over the interval
N = 80

# create mesh
mesh = IntervalMesh(N, 0, pi)

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
eigenprob = LinearEigenproblem(a, bcs=None) #bcs=bc)

# Create corresponding eigensolver, looking for n eigenvalues 
eigensolver = LinearEigensolver(eigenprob, N)#, solver_parameters={"eps_view": None})

# Solve the problem
ncov = eigensolver.solve()



def firedrake_eigenspace(k):
    eigenmodes_real, _ = eigensolver.eigenfunction2(k)
    eigenspace = eigenmodes_real.vector()[:]
    return eigenspace

def boffi_eigenspace(k, N):
    x = np.linspace(0, np.pi, N+1)
    eigenspace = np.sin(k * x)
    return eigenspace

k = 4
y_boffi = boffi_eigenspace(k, N)[1:-1] # exclude endpoints (0 and pi)
y_firedrake = firedrake_eigenspace(k)[1:-1]


# Plotting the eigenspace
x = np.linspace(0, np.pi, N+1)[1:-1]
plt.plot(x, y_boffi, label='Boffi')
plt.plot(x, y_firedrake, label='Firedrake')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.title(f'Eigenspace for k = {k}')
plt.show()