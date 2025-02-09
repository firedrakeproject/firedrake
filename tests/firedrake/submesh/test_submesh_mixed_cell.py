import os
from firedrake import *
from petsc4py import PETSc

import time
import numpy as np
import matplotlib.pyplot as plt
from firedrake.pyplot import triplot, plot, quiver
from firedrake.cython import dmcommon
from firedrake.utils import IntType
import ufl
import finat



def plot_mesh(mesh):
    if mesh.comm.size == 1:
        fig, axes = plt.subplots()
        axes.axis('equal')
        triplot(mesh, axes=axes)
        axes.legend()
        plt.savefig(mesh.name + '.pdf')
    else:
        raise RuntimeError("comm.size must be 1")


def make_mesh_netgen(h, nref, name):
    #                   labels
    #                      3
    #    +--------------------------------------+
    #    |     __ 5                             |
    #    |  6 /   \ ____11_____                 |
    #  4 |   |  12,13__________|10              | 2
    #    |  7 \__ /      9                      |
    #    |        8                             |
    #    +--------------------------------------+
    #                      1
    import netgen
    from netgen.geom2d import CSG2d, Rectangle, Circle
    geo = CSG2d()
    rect = Rectangle(pmin=(0, 0), pmax=(1, 1))
    geo.Add(rect)
    ngmesh = geo.GenerateMesh(maxh=.1, quad_dominated=True)
    #ngmesh.Export("mixed_cell_unit_square.msh","Gmsh2 Format")
    return Mesh("mixed_cell_unit_square.msh")
    #mesh.init()

dim = 2
mesh = make_mesh_netgen(0.1, 0, "a")
mesh_t = Submesh(mesh, dim, PETSc.DM.PolytopeType.TRIANGLE, label_name="celltype", name="mesh_tri")
x_t, y_t = SpatialCoordinate(mesh_t)
n_t = FacetNormal(mesh_t)
mesh_q = Submesh(mesh, dim, PETSc.DM.PolytopeType.QUADRILATERAL, label_name="celltype", name="mesh_quad")
x_q, y_q = SpatialCoordinate(mesh_q)
n_q = FacetNormal(mesh_q)
V_t = FunctionSpace(mesh_t, "P", 4)
V_q = FunctionSpace(mesh_q, "Q", 3)
#plot_mesh(mesh_t)
#plot_mesh(mesh_q)
#pgfplot(f, "mesh_tri.dat", degree=2)
dx_t = Measure("dx", mesh_t)
dx_q = Measure("dx", mesh_q)
ds_t = Measure("ds", mesh_t)
ds_q = Measure("ds", mesh_q)
# domain size
A_t = assemble(Constant(1) * dx_t)
A_q = assemble(Constant(1) * dx_q)
assert abs(A_t + A_q -1.0) < 1.e-13
HDiv_t = FunctionSpace(mesh_t, "BDM", 3)
HDiv_q = FunctionSpace(mesh_q, "RTCF", 3)
hdiv_t = Function(HDiv_t).interpolate(as_vector([x_t**2, y_t**2]))
hdiv_q = Function(HDiv_q).project(as_vector([x_q**2, y_q**2]), solver_parameters={"ksp_rtol": 1.e-13})
v_t = assemble(dot(hdiv_q, as_vector([x_q, y_q])) * ds_t(0))
v_q = assemble(dot(hdiv_t, as_vector([x_t, y_t])) * ds_q(0))
assert abs(v_q - v_t) < 1.e-13
v_t = assemble(dot(hdiv_q, as_vector([x_t, y_t])) * ds_t(0))
v_q = assemble(dot(hdiv_t, as_vector([x_q, y_q])) * ds_q(0))
assert abs(v_q - v_t) < 1.e-13
v_t = assemble(dot(hdiv_q, as_vector([x_q, y_t])) * ds_t(0))
v_q = assemble(dot(hdiv_t, as_vector([x_t, y_q])) * ds_q(0))
assert abs(v_q - v_t) < 1.e-13
v = assemble(inner(n_t, as_vector([888., 999.])) * ds_t(0))
assert abs(v) < 1.e-13
v = assemble(inner(n_q, as_vector([888., 999.])) * ds_q(0))
assert abs(v) < 1.e-13
v = assemble(inner(n_q, as_vector([888., 999.])) * ds_t(0))
assert abs(v) < 1.e-13
v = assemble(inner(n_t, as_vector([888., 999.])) * ds_q(0))
assert abs(v) < 1.e-13
v = assemble(dot(n_q + n_t, n_q + n_t) * ds_t(0))
assert abs(v) < 1.e-30
v = assemble(dot(n_q + n_t, n_q + n_t) * ds_q(0))
assert abs(v) < 1.e-30
