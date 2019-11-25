
from firedrake import *

#FEM problem (Helmholtz equation)
mesh = UnitSquareMesh(5,5)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (dot(grad(v), grad(u)) + v * u) * dx
L = f * v * dx

_A = Tensor(a)
_F = Tensor(L)
test=assemble(_A)
test2=assemble(a)
#test=assemble(_A*_F)
#test=assemble(_A*_A)
#test=assemble(-_A)
#test=assemble(Transpose(_A))

#this is getting more interesting if mixed
#b=assemble(_A.blocks[0,0])
print(test.M)
print(test2.M)