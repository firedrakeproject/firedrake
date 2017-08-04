from __future__ import absolute_import, print_function, division

import numpy as np

from firedrake import *


mesh = UnitSquareMesh(5, 5)

metric = Function(TensorFunctionSpace(mesh, 'CG', 1))
x, y = SpatialCoordinate(mesh)
metric.interpolate(as_tensor([[1+500*x, 0], [0, 1+500*y]]))

mesh._plex.view()

# test adapt function
newmesh = adapt(mesh, metric)
f = Function(VectorFunctionSpace(newmesh, 'CG', 1)).interpolate(SpatialCoordinate(newmesh))

# test adapt class

adaptor = AnisotropicAdaptation(mesh, metric)
newmesh = adaptor.adapted_mesh

newmesh._plex.view()

# test interpolation

g = Function(FunctionSpace(mesh, 'CG', 1)).interpolate(x+y)
gnew = adaptor.transfer_solution(g)[0]

xnew, ynew = SpatialCoordinate(newmesh)
hnew = Function(FunctionSpace(newmesh, 'CG', 1)).interpolate(xnew+ynew)
assert(np.allclose(gnew.dat.data, hnew.dat.data))

# test preservation of boundary labels

plex = newmesh._plex
bdLabelSize = plex.getLabelSize("Face Sets")
lis = plex.getLabelIdIS("Face Sets")
bdLabelVal = lis.getIndices()

plexnew = newmesh._plex
bdLabelSizenew = plexnew.getLabelSize("Face Sets")
assert(bdLabelSizenew == 4)
lisnew = plexnew.getLabelIdIS("Face Sets")
bdLabelValnew = lisnew.getIndices()
assert((bdLabelVal == bdLabelValnew).all)
for i in range(bdLabelSizenew):
    size = plexnew.getStratumSize("Face Sets", bdLabelValnew[i])
    assert(size > 0)
