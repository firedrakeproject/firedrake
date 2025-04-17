from firedrake import *


# Setting up mesh parameters
nx, ny = 20, 20
mesh = RectangleMesh(nx, ny, 1.0, 1.0)


# Setting up function space
degree = 4
V = FunctionSpace(mesh, "CG", degree)

# Using vertex only mesh
source_locations = [(0.5, 0.5)]
source_mesh = VertexOnlyMesh(mesh, source_locations)

print("END", flush=True)
