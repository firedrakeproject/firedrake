from firedrake import *


mesh = UnitSquareMesh(1,1)
V = FunctionSpace(mesh, "CG", 2)

u = TestFunction(V)
f = Function(V)
#g = Function(V)
f.assign(3)
#x = SpatialCoordinate(mesh)
#g.interpolate(x[0]*43)

#form = (f**2)*u*dx
form = f*u*dx
with device("gpu", [("cell", 1), ("quad", 4)]) as compute_device:
#with device("gpu", [("cell", 1)]) as compute_device:
    form_a = assemble(form)

print(form_a.dat.data)
