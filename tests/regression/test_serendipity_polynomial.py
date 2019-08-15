from firedrake import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(2, 2, quadrilateral=True)
x = SpatialCoordinate(mesh)

V1 = FunctionSpace(mesh, "CG", 1)
f = Function(V1)
g1 = Function(V1)
f.interpolate(x[1])

V2 = FunctionSpace(mesh, "S", 1)
g = Function(V2)
g.project(x[1])

#g1.project(g)

#error = sqrt(assemble(dot(f - g1, f - g1)) * dx)
#print(error)

print(f.dat.data)
print(g.dat.data)

plot(g, contour=True)
plot(f, contour=True)
plt.show()
