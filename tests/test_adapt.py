from __future__ import absolute_import, print_function, division
import numpy as np

from firedrake import *


mesh = UnitSquareMesh(5,5)

metric = Function(TensorFunctionSpace(mesh, 'CG', 1))

metric.interpolate(Constant([[100.,0],[0,100.]]))
x,y = SpatialCoordinate(mesh)
metric.interpolate(as_tensor([[1+500*x,0],[0,1+500*y]]))

newmesh = adapt(mesh, metric)


f = Function(VectorFunctionSpace(newmesh, 'CG', 1)).interpolate(SpatialCoordinate(newmesh))
File("mesha.pvd").write(f)