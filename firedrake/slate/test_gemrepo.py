
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
_A2 = Tensor(a)
_F = AssembledVector(assemble(L))

#TEST: assemble tensor
#test=assemble(_A)
#test1=assemble(_A2)
#comp=assemble(a)

#TODO: TEST: assemble coefficients
test=assemble(_F)
#comp=assemble(_L)

#TEST: assemble addition
#test=assemble(_A+_A2)
#comp=assemble(a+a)

#TEST: assemble negative
#test=assemble(-_A)
#comp=assemble(-a)

#TODO: TEST: assemble transpose
#test=assemble(Transpose(_A))

#TODO: TEST: assemble contraction
#test=assemble(_A*_A)

#TODO: TEST: assemble contraction
#test=assemble(_A*_F)

#TODO: TEST: assemble blocks
#this is getting more interesting if mixed
#b=assemble(_A.blocks[0,0])

#Test the output
print(test.M.handle.view())
print(comp.M.handle.view())

print((test.M.handle-comp.M.handle).view())
#print((test.M.handle-test3.M.handle).norm())
#print((test.M.handle-test2.M.handle).norm())
print((test.M.handle-comp.M.handle).norm())