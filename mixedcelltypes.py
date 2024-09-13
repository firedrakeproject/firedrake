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


L = 2.50
H = 0.41
Cx = 0.20  # center of hole: x
Cy = 0.20  #                 y
r = 0.05
s = r / np.sqrt(2)
x0 = 0.0
x1 = Cx - 2 * s
x2 = Cx - 1 * s
x3 = Cx + 1 * s
x4 = Cx + 2 * s
x5 = 0.6
x6 = L
y0 = 0.0
y1 = Cy - 2 * s
y2 = Cy - 1 * s
y3 = Cy + 1 * s
y4 = Cy + 2 * s
y5 = H
pointA = (0.6, 0.2)
pointB = (0.2, 0.2)
label_fluid = 1
label_struct = 2
label_left = 1
label_right = 2
label_bottom = 3
label_top = 4
label_cylinder = 5
label_interface = 6
label_cylinder_left = 11
label_cylinder_right = 12
label_cylinder_bottom = 13
label_cylinder_top = 14
label_interface_x = 15
label_interface_y = 16
label_struct_base = 20


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
    from netgen.geom2d import CSG2d, Rectangle, Circle,SplineGeometry
    geo = CSG2d()
    rect0 = Rectangle(pmin=(0, 0), pmax=(1, 1))
    rect1 = Rectangle(pmin=(-1, -1), pmax=(2, 2))
    #circ0 = Rectangle(pmin=(x2, y2), pmax=(x3, y3))
    #circ1 = Rectangle(pmin=(x1, y1), pmax=(x4, y4))
    #fluid_struct = rect0 - circ0
    #fluid = fluid_struct - rect1
    #struct = fluid_struct * rect1
    geo.Add(rect1 - rect0)
    #geo.Add(fluid - fluid * circ1)
    geo.Add(rect0)
    #geo.Add(struct - struct * circ1)

    """
    rect = Rectangle(pmin=(0, 0), pmax=(1, 1), left="line", right="line", bottom="line", top="line")
    geo.Add(rect)
    """
    """
    geo = SplineGeometry()
    pnts = [(0, 0), (1, 0), (1, 1),
            (0, 1)]
    p1, p2, p3, p4 = [geo.AppendPoint(*pnt) for pnt in pnts]
    curves = [[["line", p1, p2], "line"],
              [["line", p2, p3], "line"],
              [["line", p3, p4], "line"],
              [["line", p4, p1], "line"]]
    [geo.Append(c, bc=bc) for c, bc in curves]
    """
    ngmesh = geo.GenerateMesh(maxh=.1, quad_dominated=True)
    #ngmesh.Export("mixed_cell_unit_square.msh","Gmsh2 Format")
    #raise RuntimeError
    return Mesh("mixed_cell_unit_square.msh")
    #mesh.init()


dim = 2
#mesh = Mesh(os.path.join(os.environ.get("PETSC_DIR"), "share/petsc/datafiles/meshes/hybrid_triquad.msh"))
mesh = make_mesh_netgen(0.1, 0, "a")
mesh = Submesh(mesh, dim, 2)
mesh.topology_dm.viewFromOptions("-dm_view")
mesh_t = Submesh(mesh, dim, PETSc.DM.PolytopeType.TRIANGLE, label_name="celltype", name="mesh_tri")
x_t, y_t = SpatialCoordinate(mesh_t)
n_t = FacetNormal(mesh_t)
mesh_q = Submesh(mesh, dim, PETSc.DM.PolytopeType.QUADRILATERAL, label_name="celltype", name="mesh_quad")
x_q, y_q = SpatialCoordinate(mesh_q)
plot_mesh(mesh_t)
plot_mesh(mesh_q)
raise RuntimeError
n_q = FacetNormal(mesh_q)
V_t = FunctionSpace(mesh_t, "P", 4)
V_q = FunctionSpace(mesh_q, "Q", 3)
v_t = TestFunction(V_t)
u_t = TrialFunction(V_t)
v_q = TestFunction(V_q)
u_q = TrialFunction(V_q)

dx_t = Measure("dx", mesh_t)
dx_q = Measure("dx", mesh_q)
ds_t = Measure("ds", mesh_t)
ds_q = Measure("ds", mesh_q)

g_t = Function(V_t).interpolate(sin(y_t))
#g_q = Constant(1)
v1 = assemble(Constant(1) * ds_q(subdomain_id="otherwise"))
print(v1)
raise RuntimeError






a = inner(grad(u_t), grad(v_t)) * dx_t + \
    inner(grad(u_q), grad(v_q)) * dx_q

f = Function(V)
f.assign(0)
L = inner(f, v) * dx

# This value of the stabilisation parameter gets us about 4 sf
# accuracy.
h = 0.25
gamma = 0.00001

n = FacetNormal(mesh)

B = a - \
    inner(dot(grad(u), n), v)*(ds(3) + ds(4)) - \
    inner(u, dot(grad(v), n))*(ds(3) + ds(4)) + \
    (1.0/(h*gamma))*inner(u, v)*(ds(3) + ds(4))

u_0 = Function(V)
u_0.assign(0)
u_1 = Function(V)
u_1.assign(42)

F = L - \
    inner(u_0, dot(grad(v), n)) * ds(3) - \
    inner(u_1, dot(grad(v), n)) * ds(4) + \
    (1.0/(h*gamma))*inner(u_0, v) * ds(3) + \
    (1.0/(h*gamma))*inner(u_1, v) * ds(4)

u = Function(V)
solve(B == F, u)

f = Function(V)
x = SpatialCoordinate(mesh)
f.interpolate(42*x[1])
