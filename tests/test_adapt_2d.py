from __future__ import absolute_import, print_function, division

import numpy as np
from os.path import abspath, join, dirname

from firedrake import *


mesh = UnitSquareMesh(5, 5)

metric = Function(TensorFunctionSpace(mesh, 'CG', 1))
x, y = SpatialCoordinate(mesh)
metric.interpolate(as_tensor([[1+500*x, 0], [0, 1+500*y]]))

# test adapt function

newmesh = adapt(mesh, metric)
f = Function(VectorFunctionSpace(newmesh, 'CG', 1)).interpolate(SpatialCoordinate(newmesh))

# test adapt class

adaptor = AnisotropicAdaptation(mesh, metric)
newmesh = adaptor.adapted_mesh
num_vertices_mesh1 = newmesh.num_vertices()

# test interpolation

g = Function(FunctionSpace(mesh, 'CG', 1)).interpolate(x+y)
gnew = adaptor.transfer_solution(g)[0]

xnew, ynew = SpatialCoordinate(newmesh)
hnew = Function(FunctionSpace(newmesh, 'CG', 1)).interpolate(xnew+ynew)
assert(np.allclose(gnew.dat.data, hnew.dat.data))

# test preservation of boundary labels

plex = mesh._topology_dm
bdLabelSize = plex.getLabelSize("Face Sets")
lis = plex.getLabelIdIS("Face Sets")
bdLabelVal = lis.getIndices()

plexnew = newmesh._topology_dm
bdLabelSizenew = plexnew.getLabelSize("Face Sets")
assert(bdLabelSizenew == bdLabelSize)
lisnew = plexnew.getLabelIdIS("Face Sets")
bdLabelValnew = lisnew.getIndices()
assert((bdLabelVal == bdLabelValnew).all)
for i in range(bdLabelSizenew):
    size = plexnew.getStratumSize("Face Sets", bdLabelValnew[i])
    assert(size > 0)

# test preservation of cell labels

cwd = abspath(dirname(__file__))
mesh = Mesh(join(cwd, "meshes", "circle_in_square.msh"))
Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
f = Function(Vc).interpolate(as_vector([(x+2)*0.25, (y+2)*0.25]))
mesh.coordinates.assign(f)

metric = Function(TensorFunctionSpace(mesh, 'CG', 1))
x, y = SpatialCoordinate(mesh)
metric.interpolate(as_tensor([[1+500*x, 0], [0, 1+500*y]]))

adaptor = AnisotropicAdaptation(mesh, metric)
newmesh = adaptor.adapted_mesh
num_vertices_mesh2 = newmesh.num_vertices()
assert(abs(num_vertices_mesh2 - num_vertices_mesh1) < 0.05*num_vertices_mesh1)
_ = FunctionSpace(newmesh, "CG", 1)  # Make something on the mesh

for i in [3, 4]:
    assert len(mesh.cell_subset(i).indices) > 0
    assert len(newmesh.cell_subset(i).indices) > 0
assert len(mesh.exterior_facets.unique_markers) > 0
assert len(newmesh.exterior_facets.unique_markers) > 0
assert np.allclose(mesh.exterior_facets.unique_markers, newmesh.exterior_facets.unique_markers)
