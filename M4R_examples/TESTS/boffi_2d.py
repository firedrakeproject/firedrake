
"""This demo program solves the Poisson eigenvalue problem

  - div grad u(x) = lambda u(x) (STRONG FORM)
  grad u(x) dot grad v(x) dx = lambda u(x)v(x) dx (WEAK FORM)

on the 2d domain  (0, pi) x (0, pi) 

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi) TO CHANGE
"""
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# number of elements in each direction
n = 5 # 10 x 10 elements

# create mesh
mesh = RectangleMesh(n, n, pi, pi, quadrilateral=False) 

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

# Create corresponding eigensolver, looking for n_evals eigenvalues
n_evals = 15
eigensolver = LinearEigensolver(eigenprob, n_evals)

# Solve the problem
ncov = eigensolver.solve()

def evals():
  for k in range(0, ncov):
      eval = eigensolver.eigenvalue(k)
      print(1/eval)

# for k in range(ncov):
#     eigenmodes_real, _ = eigensolver.eigenfunction(k)
#     # eigenmodes.dat.data[:]
#     evec = eigenmodes_real.vector()[1:-1] # scale by max norm
#     max_evec_val = max(evec)
#     evec_scaled = evec / max_evec_val
# '''TESTING THE EFUNCTIONS'''

x, y = SpatialCoordinate(mesh)

def mesh(n):
    mesh = RectangleMesh(n, n, pi, pi, quadrilateral=False) 
    fig, axes = plt.subplots()
    triplot(mesh, axes=axes, interior_kw={'linewidths':(0.2)})
    axes.set_title(f'N = {n} mesh')
    plt.show()
#mesh(3)
def graph_analytical_solns(m, n):
    e_analytic = interpolate(sin(m*x)*sin(n*y), V)
    fig, axes = plt.subplots()
    colors = tripcolor(e_analytic, axes=axes)
    triplot(mesh, axes=axes, interior_kw={'linewidths':(0.2)})
    #fig.colorbar(colors)
    axes.set_title(f'Analytical solution of eigenfunction, m={m}, n={n}')
    plt.show()
  
graph_analytical_solns(2, 2)
# # Plot the tripcolor
# plt.tripcolor(mesh.coordinates.dat[:, 0], mesh.coordinates.dat[:, 1],
#               mesh.cells(), u.dat.data, shading="gouraud")

# Overlay the mesh
plt.triplot(mesh.coordinates.dat[:, 0], mesh.coordinates.dat[:, 1], mesh.cells(), color="black")

def graph_eigensolver_solns(i):
   
    fig, axes = plt.subplots()
    real, imag = eigensolver.eigenfunction(i) # take real part
    real /= max(real.dat.data[:])
    color1 = tripcolor(real, axes=axes)
    fig.colorbar(color1)
    axes.set_title(f'Eigensolver approximation of eigenfunctions, k={i}')
    plt.show()


for i in range(6):
    graph_eigensolver_solns(i)
    graph_analytical_solns(i+1, i+1)



# def graph_both(k):
#     fig, axes = plt.subplots(k, 2, figsize=(12, 8))
#     for i in range(1, k):
        
#         e_analytic = interpolate(sin(m*x)*sin(n*y), V)
#         colors = tripcolor(e_analytic, axes=axes[i,0])

#         real, imag = eigensolver.eigenfunction(i-1)
#         eigensolver_efunc = tripcolor(real, axes=axes[i, 1])
#         axes[i,1].set_title('Eigensolver approximation of eigenfunctions')

    # plt.show()
