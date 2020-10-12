from firedrake import *

base_mesh = UnitIntervalMesh(10)
mesh = ExtrudedMesh(base_mesh, layers=4, layer_height=1.0/10.0)

Q1D = FunctionSpace(base_mesh, 'CG', 1)
x = SpatialCoordinate(base_mesh)
f1D = Function(Q1D).interpolate(cos(2.0 * pi* x[0]))
fextended = ExtrudedExtendFunction(mesh, base_mesh, f1D)
fextended.rename('fextended')
File('fextended-None.pvd').write(fextended)

x,y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, 'CG', 1)
foriginal = Function(V).interpolate(cos(2.0 * pi* x) * y)
foriginal.rename('foriginal')
fbottom = ExtrudedExtendFunction(mesh, base_mesh, foriginal, extend_type='bottom')
fbottom.rename('fbottom')
ftop = ExtrudedExtendFunction(mesh, base_mesh, foriginal, extend_type='top')
ftop.rename('ftop')
File('fextended-topbottom.pvd').write(foriginal,ftop,fbottom)

