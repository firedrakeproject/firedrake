import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
'''
Wrote this preFriday
'''

# Standard
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1) # bc
eigenmodes = Function(V)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)


num_eigenvalues = 1

# to change
func = sin(x)
f.interpolate(func) 

# after multiplying with test fn
LHS = inner(inner(u, grad(u)), v)
RHS = inner(f, v)
LHS_integral =  LHS * dx
RHS_integral = RHS * dx

# building solver
u2 = Function(V)

params = {'ksp_type': 'cg', 'pc_type': 'none'}
solve(LHS_integral == RHS_integral, u2, solver_parameters=params)

# view

File("eigen.pvd").write(u2)

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

try:
    fig, axes = plt.subplots()
    colors = tripcolor(u, axes=axes)
    fig.colorbar(colors)
except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e)

try:
    fig, axes = plt.subplots()
    contours = tricontour(u, axes=axes)
    fig.colorbar(contours)
except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e)

try:
    plt.show()
except Exception as e:
    warning("Cannot show figure. Error msg: '%s'" % e)

# check
f.interpolate(cos(x*pi*2)*cos(y*pi*2))
print(sqrt(assemble(dot(u - f, u - f) * dx)))