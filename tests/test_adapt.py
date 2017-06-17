from __future__ import absolute_import, print_function, division

from firedrake import *


mesh = UnitSquareMesh(5, 5)

metric = Function(TensorFunctionSpace(mesh, 'CG', 1))
x, y = SpatialCoordinate(mesh)
metric.interpolate(as_tensor([[1+500*x, 0], [0, 1+500*y]]))

# test adapt function
newmesh = adapt(mesh, metric)
f = Function(VectorFunctionSpace(newmesh, 'CG', 1)).interpolate(SpatialCoordinate(newmesh))
File("mesha.pvd").write(f)

# test adapt class

adaptor = AAdaptation(mesh, metric)
newmesh = adaptor.newmesh

g = Function(FunctionSpace(mesh, 'CG', 1))
g.interpolate(x+y)
gnew = adaptor.transfer_solution(g)[0]

File("mesha2.pvd").write(gnew)

# test preservation of boundary labels

plex = newmesh._plex
bdLabelSize = plex.getLabelSize("boundary_ids")
lis = plex.getLabelIdIS("boundary_ids")
bdLabelVal = lis.getIndices()

plexnew = newmesh._plex
bdLabelSizenew = plexnew.getLabelSize("boundary_ids")
assert(bdLabelSizenew==4)
lisnew = plexnew.getLabelIdIS("boundary_ids")
bdLabelValnew = lisnew.getIndices()
assert((bdLabelVal==bdLabelValnew).all)
for i in range(bdLabelSizenew):
    size = plexnew.getStratumSize("boundary_ids", bdLabelValnew[i])
    assert(size > 0)
