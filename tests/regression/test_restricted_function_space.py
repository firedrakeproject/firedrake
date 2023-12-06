import pytest
from firedrake import *
import numpy as np

mesh = UnitSquareMesh(1, 1)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)

u = TrialFunction(V)
v = TestFunction(V)

matrix = assemble(u * v * dx)
print(matrix.M.values)

bc = DirichletBC(V, 0, "1")
V_res = RestrictedFunctionSpace(V, name="Restricted", bcs=[bc]) # currently fails here from caching

u2 = TrialFunction(V_res)
v2 = TestFunction(V_res)

matrix_res = assemble(u2 * v2 * dx)
print(matrix_res.M.values)
