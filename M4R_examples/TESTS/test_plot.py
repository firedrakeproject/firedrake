from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
x = SpatialCoordinate(mesh)
u.interpolate(x[0] + x[1])
fig, axes = plt.subplots()
levels = np.linspace(0, 1, 51)
contours = tricontourf(u, levels=levels, axes=axes, cmap="inferno")
axes.set_aspect("equal")
fig.colorbar(contours)
fig.show()
