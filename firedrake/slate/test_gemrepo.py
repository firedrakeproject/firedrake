
from firedrake import *

#FEM problem (Helmholtz equation)
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (dot(grad(v), grad(u)) + v * u) * dx
L = f * v * dx

#Element tensors defining the local 3-by-3 block system
_A = Tensor(a)
_F = Tensor(L)
#assemble(_A)
assemble(_A+_A)