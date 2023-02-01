import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)


mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1) # linear polynomials therefore 1 in third arg
bc = DirichletBC(V, 0.0, "on_boundary") # homogenous Dirichlet boundaries   

u, v = TrialFunction(V), TestFunction(V)

# weak form of the Helmholtz equation
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
f = Function(V)
L = inner(f, v) * dx
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

A, b = assemble(a, L, bc)
A = as_backend_type(A).mat() # convert the firedrake Matrix to a petsc4py.PETSc.Mat
b = as_backend_type(b).vec() # convert the firedrake Vector to a petsc4py.PETSc.Vec
eigensolver = SLEPc.EPS().create()
eigensolver.setOperators(A,b)
eigensolver.solve()
evals = eigensolver.getEigenvalues()
evecs = eigensolver.getEigenvectors()
print(evals)