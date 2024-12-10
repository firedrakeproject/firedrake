import os
import time
import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import triplot, plot, quiver
from firedrake.cython import dmcommon
from firedrake.utils import IntType
import ufl
import finat
import tracemalloc
import gc


tracemalloc.start()



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
    from netgen.geom2d import CSG2d, Rectangle, Circle
    geo = CSG2d()
    rect0 = Rectangle(pmin=(0, 0), pmax=(L, H))
    circ0 = Rectangle(pmin=(x2, y2), pmax=(x3, y3))
    circ1 = Rectangle(pmin=(x1, y1), pmax=(x4, y4))
    rect1 = Rectangle(pmin=(Cx, 0.19), pmax=(x5, 0.21))
    fluid_struct = rect0 - circ0
    fluid = fluid_struct - rect1
    struct = fluid_struct * rect1
    geo.Add(fluid * circ1)
    geo.Add(fluid - fluid * circ1)
    geo.Add(struct * circ1)
    geo.Add(struct - struct * circ1)
    ngmesh = geo.GenerateMesh(maxh=h)
    mesh = Mesh(ngmesh)
    if nref > 0:
        mesh = MeshHierarchy(mesh, nref, netgen_flags={"degree": 1})[-1]
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "DP", 0)
    c0 = Function(V)  # to empty
    c1 = Function(V).interpolate(conditional(Or(x < x3, x > x5), 1.,
                                 conditional(Or(y < 0.19, y > 0.21), 1., 0.)))
    c2 = Function(V).interpolate(conditional(Or(x < x3, x > x5), 0.,
                                 conditional(Or(y < 0.19, y > 0.21), 0., 1.)))
    V = FunctionSpace(mesh, "HDiv Trace", 0)
    f0 = Function(V)  # to empty labels 8, 9, 10, 11, 12, 21
    f1 = Function(V)
    DirichletBC(V, 1., (16, )).apply(f1)
    f2 = Function(V)
    DirichletBC(V, 1., (14, )).apply(f2)
    f3 = Function(V)
    DirichletBC(V, 1., (13, )).apply(f3)
    f4 = Function(V)
    DirichletBC(V, 1., (15, )).apply(f4)
    f5 = Function(V)
    DirichletBC(V, 1., (2, 3, 4, 5, 6, 20)).apply(f5)
    f6 = Function(V)
    DirichletBC(V, 1., (1, 7, 17, 18, 19)).apply(f6)
    f11 = Function(V)
    DirichletBC(V, 1., (4, )).apply(f11)
    f12 = Function(V)
    DirichletBC(V, 1., (2, 6, 20)).apply(f12)
    f13 = Function(V)
    DirichletBC(V, 1., (5, )).apply(f13)
    f14 = Function(V)
    DirichletBC(V, 1., (3, )).apply(f14)
    f15 = Function(V)
    DirichletBC(V, 1., (1, 7, 17, 19)).apply(f15)
    f16 = Function(V)
    DirichletBC(V, 1., (18, )).apply(f16)
    f20 = Function(V)
    DirichletBC(V, 1., (20, )).apply(f20)
    return RelabeledMesh(mesh, [c0 for i in range(1, 5)] + [c1, c2] + [f0 for i in range(1, 22)] + [f1, f2, f3, f4, f5, f6] + [f11, f12, f13, f14, f15, f16, f20], [i for i in range(1, 5)] + [label_fluid, label_struct] + [i for i in range(1, 22)] + [label_left, label_right, label_bottom, label_top, label_cylinder, label_interface] + [label_cylinder_left, label_cylinder_right, label_cylinder_bottom, label_cylinder_top, label_interface_x, label_interface_y, label_struct_base], name=name)


def _finalise_mesh(mesh, degree):
    V = VectorFunctionSpace(mesh, "CG", degree)
    f = Function(V).interpolate(mesh.coordinates)
    """
    x = f.dat.data_with_halos[:, 0]
    y = f.dat.data_with_halos[:, 1]
    eps = 1.e-6
    cond = ((x > x1 - eps) & (x < x2 + eps) & (y > y2 - eps) & (y < y3 + eps))
    x[cond] = x[cond] - (x[cond] - x1) / (x2 - x1) * (np.sqrt(r ** 2 - (y[cond] - Cy) ** 2) - s)
    cond = ((x > x3 - eps) & (x < x4 + eps) & (y > y2 - eps) & (y < y3 + eps))
    x[cond] = x[cond] + (x[cond] - x4) / (x3 - x4) * (np.sqrt(r ** 2 - (y[cond] - Cy) ** 2) - s)
    cond = ((x > x2 - eps) & (x < x3 + eps) & (y > y1 - eps) & (y < y2 + eps))
    y[cond] = y[cond] - (y[cond] - y1) / (y2 - y1) * (np.sqrt(r ** 2 - (x[cond] - Cx) ** 2) - s)
    cond = ((x > x2 - eps) & (x < x3 + eps) & (y > y3 - eps) & (y < y4 + eps))
    y[cond] = y[cond] + (y[cond] - y4) / (y3 - y4) * (np.sqrt(r ** 2 - (x[cond] - Cx) ** 2) - s)
    """
    return Mesh(f, name=mesh.name)


def _mesh_displacement(V):
    f = Function(V)
    fx = f.dat.data_with_halos[:, 0]
    fy = f.dat.data_with_halos[:, 1]
    coordinates = Function(V).interpolate(V.mesh().coordinates)
    x = coordinates.dat.data_with_halos[:, 0]
    y = coordinates.dat.data_with_halos[:, 1]
    eps = 1.e-6
    cond = ((x > x1 - eps) & (x < x2 + eps) & (y > y2 - eps) & (y < y3 + eps))
    fx[cond] = - (x[cond] - x1) / (x2 - x1) * (np.sqrt(r ** 2 - (y[cond] - Cy) ** 2) - s)
    cond = ((x > x3 - eps) & (x < x4 + eps) & (y > y2 - eps) & (y < y3 + eps))
    fx[cond] = (x[cond] - x4) / (x3 - x4) * (np.sqrt(r ** 2 - (y[cond] - Cy) ** 2) - s)
    cond = ((x > x2 - eps) & (x < x3 + eps) & (y > y1 - eps) & (y < y2 + eps))
    fy[cond] = - (y[cond] - y1) / (y2 - y1) * (np.sqrt(r ** 2 - (x[cond] - Cx) ** 2) - s)
    cond = ((x > x2 - eps) & (x < x3 + eps) & (y > y3 - eps) & (y < y4 + eps))
    fy[cond] = (y[cond] - y4) / (y3 - y4) * (np.sqrt(r ** 2 - (x[cond] - Cx) ** 2) - s)
    return f


class DirichletBCArgyrisNoSlip(DirichletBC):
    def __init__(self, V):
        assert isinstance(V.finat_element, finat.Argyris)
        assert V.ufl_element().degree() == 5
        super().__init__(V, Constant((0., 0.)), None)
        plex = V.mesh().topology_dm
        sec = V.dm.getLocalSection()
        label = plex.getLabel("Face Sets")
        nodes = []
        for label_value, axis in [(label_left, 'y'),
                                  (label_right, 'y'),
                                  (label_bottom, 'x'),
                                  (label_top, 'x'),
                                  (label_cylinder_left, 'y'),
                                  (label_cylinder_right, 'y'),
                                  (label_cylinder_bottom, 'x'),
                                  (label_cylinder_top, 'x')]:
            local_dof_ids = {'x': [0, 1, 3],
                             'y': [0, 2, 5]}[axis]
            label_is_size = label.getStratumSize(label_value)
            if label_is_size > 0:
                label_is = label.getStratumIS(label_value)
                with label_is as points:
                    for p in points:
                        dof = sec.getDof(p)
                        off = sec.getOffset(p)
                        if dof == 0:  # cell
                            pass
                        elif dof == 1:  # facet
                            pass
                        elif dof == 6:  # vertex
                            nodes.extend([off + l for l in local_dof_ids])
                        else:
                            raise RuntimeError
        self.nodes = np.unique(np.array(nodes, dtype=IntType))


class DirichletBCEquationBCArgyrisNoSlip(DirichletBC):
    def __init__(self, V):
        assert isinstance(V.finat_element, finat.Argyris)
        assert V.ufl_element().degree() == 5
        super().__init__(V, Constant((0., 0.)), None)
        plex = V.mesh().topology_dm
        sec = V.dm.getLocalSection()
        label = plex.getLabel("Face Sets")
        nodes = []
        label_is_size = label.getStratumSize(label_struct_base)
        if label_is_size > 0:
            assert label_is_size in [1, 2], "There are only two vertices at intersection of cylinder and structure"
            label_is = label.getStratumIS(label_struct_base)
            with label_is as points:
                for p in points:
                    dof = sec.getDof(p)
                    off = sec.getOffset(p)
                    if dof == 0:  # cell
                        raise RuntimeError
                    elif dof == 1:  # facet
                        raise RuntimeError
                    elif dof == 6:  # vertex
                        nodes.extend([off + 0])
                    else:
                        raise RuntimeError
        self.nodes = np.unique(np.array(nodes, dtype=IntType))


class EquationBCArgyrisNoSlip(EquationBC):
    def __init__(self, equation, solution, V):
        assert isinstance(V.finat_element, finat.Argyris)
        assert V.ufl_element().degree() == 5
        bbc = DirichletBCEquationBCArgyrisNoSlip(V)
        super().__init__(equation, solution, None, bcs=[bbc], V=V)
        plex = V.mesh().topology_dm
        sec = V.dm.getLocalSection()
        label = plex.getLabel("Face Sets")
        nodes = []
        for label_value, axis in [(label_interface_x, 'x'),
                                  (label_interface_y, 'y')]:
            local_dof_ids = {'x': [0, 1, 3],
                             'y': [0, 2, 5]}[axis]
            label_is_size = label.getStratumSize(label_value)
            if label_is_size > 0:
                label_is = label.getStratumIS(label_value)
                with label_is as points:
                    for p in points:
                        dof = sec.getDof(p)
                        off = sec.getOffset(p)
                        if dof == 0:  # cell
                            pass
                        elif dof == 1:  # facet
                            pass
                        elif dof == 6:  # vertex
                            nodes.extend([off + l for l in local_dof_ids])
                        else:
                            raise RuntimeError(f"{dof}")
        nodes = np.unique(np.array(nodes, dtype=IntType))
        self._F.nodes = nodes
        self._J.nodes = nodes
        self._Jp.nodes = nodes


dim = 2
degree = 5
nref = 3
nquad = 16
mesh  = make_mesh_netgen(0.11, nref, "mesh_fluid_struct")
mesh_f = Submesh(mesh, dim, label_fluid, name="mesh_fluid")
mesh_s = Submesh(mesh, dim, label_struct, name="mesh_struct")
#mesh = _finalise_mesh(mesh, degree)
#mesh_f = _finalise_mesh(mesh_f, degree)
#mesh_s = _finalise_mesh(mesh_s, degree)
#plot_mesh(mesh)
#plot_mesh(mesh_f)
#plot_mesh(mesh_s)
#raise RuntimeError("Just plotted.")
x, y = SpatialCoordinate(mesh)
x_f, y_f = SpatialCoordinate(mesh_f)
x_s, y_s = SpatialCoordinate(mesh_s)
n_f = FacetNormal(mesh_f)
n_s = FacetNormal(mesh_s)
dx_f = Measure(
    "dx", domain=mesh_f,
)
dx_s = Measure(
    "dx", domain=mesh_s,
)
ds_f_only = Measure("ds", domain=mesh_f)
ds_f_interf = Measure(
    "ds", domain=mesh_f,
    extra_measures=(
        Measure("ds", mesh_s),
    ),
)
ds_s_interf = Measure(
    "ds", domain=mesh_s,
    extra_measures=(
        Measure("ds", mesh_f),
    ),
)
if False:#mesh.comm.size == 1:
    fig, axes = plt.subplots()
    axes.axis('equal')
    #quiver(solution.subfunctions[0])
    triplot(mesh, axes=axes)
    axes.legend()
    plt.savefig('mesh_orig.pdf')
    fig, axes = plt.subplots()
    axes.axis('equal')
    #quiver(solution.subfunctions[0])
    triplot(mesh_f, axes=axes)
    axes.legend()
    plt.savefig('mesh_f.pdf')
    fig, axes = plt.subplots()
    axes.axis('equal')
    #quiver(solution.subfunctions[0])
    triplot(mesh_s, axes=axes)
    axes.legend()
    plt.savefig('mesh_s.pdf')
    raise RuntimeError("not error")
T = 20
dt = Constant(0.001)
chkstride = int(0.01 / float(dt))  # save every 0.01 sec
t = Constant(0.0)
CNshift = 1
mmt = "biharmonic"
fname_checkpoint = f"dumbdata/fsi3_P{degree}_P{degree - 2}_Argyris5_nref{nref}_{float(dt):.3f}_shift{CNshift}_{mmt}_temp"
fname_FD = f"dumbdata/time_series_FD_P{degree}_P{degree - 2}_Argyris5_nref{nref}_{float(dt):.3f}_shift{CNshift}_{mmt}_temp.dat"
fname_FL = f"dumbdata/time_series_FL_P{degree}_P{degree - 2}_Argyris5_nref{nref}_{float(dt):.3f}_shift{CNshift}_{mmt}_temp.dat"
fname_ux = f"dumbdata/time_series_ux_P{degree}_P{degree - 2}_Argyris5_nref{nref}_{float(dt):.3f}_shift{CNshift}_{mmt}_temp.dat"
fname_uy = f"dumbdata/time_series_uy_P{degree}_P{degree - 2}_Argyris5_nref{nref}_{float(dt):.3f}_shift{CNshift}_{mmt}_temp.dat"
rho_s = 1. * 1.e+3
nu_s = 0.4
mu_s = 2.0 * 1.e+6
rho_f = 1.e+3
nu_f = 1.e-3
Ubar = 2.0
# Re = 200.
g_s = Constant(0.0)
E_s = mu_s * 2 * (1 + nu_s)
lambda_s = nu_s * E_s / (1 + nu_s) / (1 - 2 * nu_s)
# ALE constants
nu_ale = Constant(float(nu_s))
mu_ale = Constant(float(mu_s))
E_ale = mu_ale * 2 * (1 + nu_ale)
lambda_ale = nu_ale * E_ale / (1 + nu_ale) / (1 - 2 * nu_ale)
V_0 = VectorFunctionSpace(mesh_f, "P", degree)
V_1 = VectorFunctionSpace(mesh_s, "Argyris", degree)
V_2 = VectorFunctionSpace(mesh_f, "Argyris", degree)
V_3 = VectorFunctionSpace(mesh_s, "Argyris", degree)
V_4 = FunctionSpace(mesh_f, "P", degree - 2)
V = V_0 * V_1 * V_2 * V_3 * V_4
solution = Function(V)
solution_0 = Function(V)
v_f, v_s, u_f, u_s, p = split(solution)
v_f_0, v_s_0, u_f_0, u_s_0, p_0 = split(solution_0)
dv_f, dv_s, du_f, du_s, dp = split(TestFunction(V))
for subf, name in zip(solution.subfunctions, ["v_f", "v_s", "u_f", "u_s", "p"]):
    subf.rename(name)
#
V_f_m = VectorFunctionSpace(mesh_f, "P", degree)
u_f_m = _mesh_displacement(V_f_m)
V_s_m = VectorFunctionSpace(mesh_s, "P", degree)
u_s_m = _mesh_displacement(V_s_m)
def compute_elast_tensors(dim, u, lambda_s, mu_s, u_m):
    F_m = Identity(dim) + grad(u_m)
    J_m = det(F_m)
    F = Identity(dim) + dot(grad(u), inv(F_m))
    J = det(F)
    E = 1. / 2. * (dot(transpose(F), F) - Identity(dim))
    S = lambda_s * tr(E) * Identity(dim) + 2.0 * mu_s * E
    return F, J, E, S, F_m, J_m
theta_p = Constant(1. / 2. + CNshift * float(dt))
theta_m = Constant(1. / 2. - CNshift * float(dt))
v_f_dot = (v_f - v_f_0) / dt
u_f_dot = (u_f - u_f_0) / dt
v_s_dot = (v_s - v_s_0) / dt
u_s_dot = (u_s - u_s_0) / dt
def _fluid(v_f, u_f, p, u_f_m):
    F_f, J_f, E_f, S_f, F_f_m, J_f_m = compute_elast_tensors(dim, u_f, lambda_ale, mu_ale, u_f_m)
    if mmt == "laplacian":
        mmt_domain_eq = J_f_m * inner(dot(grad(u_f), inv(F_f_m)), dot(grad(du_f), inv(F_f_m)))
    elif mmt == "elast":
        epsilon = sym(dot(grad(u_f), inv(F_f_m)))
        sigma = 2 * mu_ale * epsilon + lambda_ale * tr(epsilon) * Identity(dim)
        mmt_domain_eq = J_f_m * inner(sigma, dot(grad(du_f), inv(F_f_m)))
    elif mmt == "biharmonic":
        def grad_(u_):
            return dot(grad(u_), inv(F_f_m))
        def tr3(tensor_):
            ii, jj, kk = ufl.indices(3)
            part = ufl.classes.ComponentTensor(tensor_[ii, jj, kk], ufl.classes.MultiIndex((jj, kk)))
            return ufl.classes.ComponentTensor(tr(part), ufl.classes.MultiIndex((ii, )))
        mmt_domain_eq = J_f_m * inner(tr3(grad_(grad_(u_f))), tr3(grad_(grad_(du_f))))
    else:
        raise NotImplementedError(f"Unknown mmt : {mmt}")
    return (inner(rho_f * J_f * J_f_m * v_f_dot, dv_f) +
            inner(rho_f * J_f * J_f_m * dot(dot(dot(grad(v_f), inv(F_f_m)), inv(F_f)), v_f - u_f_dot), dv_f) +
            inner(rho_f * J_f * J_f_m * nu_f * 2 * sym(dot(dot(grad(v_f), inv(F_f_m)), inv(F_f))), dot(dot(grad(dv_f), inv(F_f_m)), inv(F_f))) -
            J_f * J_f_m * inner(p, tr(dot(dot(grad(dv_f), inv(F_f_m)), inv(F_f)))) +
            J_f * J_f_m * inner(tr(dot(dot(grad(v_f), inv(F_f_m)), inv(F_f))), dp) +
            mmt_domain_eq) * dx_f(degree=nquad)
def _struct(v_f, u_f, p, v_s, u_s, u_f_m, u_s_m):
    F_f, J_f, E_f, S_f, F_f_m, J_f_m = compute_elast_tensors(dim, u_f, lambda_s, mu_s, u_f_m)
    F_s, J_s, E_s, S_s, F_s_m, J_s_m = compute_elast_tensors(dim, u_s, lambda_s, mu_s, u_s_m)
    return (inner(rho_s * J_s * J_s_m * v_s_dot, dv_s) +
            inner(J_s_m * dot(F_s, S_s), dot(grad(dv_s), inv(F_s_m))) -
            inner(rho_s * J_s * J_s_m * as_vector([0., - g_s]), dv_s) +
            inner(J_s * J_s_m * (u_s_dot - v_s), du_s)) * dx_s(degree=nquad) + \
           inner(dot(- p * Identity(dim) + rho_f * nu_f * 2 * sym(dot(dot(grad(v_f), inv(F_f_m)), inv(F_f))), dot(dot(J_f * transpose(inv(F_f)), J_f_m * transpose(inv(F_f_m))), n_f)), dv_s) * ds_s_interf(label_interface, degree=nquad)
residual_f = theta_p * _fluid(v_f, u_f, p, u_f_m) + \
             theta_m * _fluid(v_f_0, u_f_0, p_0, u_f_m)
residual_s = theta_p * _struct(v_f, u_f, p, v_s, u_s, u_f_m, u_s_m) + \
             theta_m * _struct(v_f_0, u_f_0, p_0, v_s_0, u_s_0, u_f_m, u_s_m)
residual = residual_f + residual_s
def v_f_left(t_):
    return 1.5 * Ubar * y_f * (H - y_f) / ((H / 2) ** 2) * conditional(t_ < 2.0 + dt / 10., (1 - cos(pi / 2 * t_)) / 2., 1.)
bc_v_f_inflow = DirichletBC(V.sub(0), as_vector([theta_p * v_f_left(t - dt + theta_p * dt) +
                                                 theta_m * v_f_left(t - dt + theta_m * dt), 0.]), (label_left, ))
bc_v_f_zero = DirichletBC(V.sub(0), Constant((0, 0)), (label_bottom, label_top, label_cylinder))
bbc_v_f_noslip = DirichletBC(V.sub(0), Constant((0, 0)), ((label_cylinder, label_interface), ))
bc_v_f_noslip = EquationBC(inner(v_f - v_s, dv_f) * ds_f_interf(label_interface) == 0, solution, label_interface, bcs=[bbc_v_f_noslip], V=V.sub(0))
bc_v_s_zero = DirichletBCArgyrisNoSlip(V.sub(1))
bc_u_f_zero = DirichletBCArgyrisNoSlip(V.sub(2))
bc_u_f_noslip = EquationBCArgyrisNoSlip(inner(u_f - u_s, du_f) * ds_f_interf(label_interface) == 0, solution, V.sub(2))
bc_u_s_zero = DirichletBCArgyrisNoSlip(V.sub(3))

solver_parameters = {
    "mat_type": "aij",
    "snes_mat_it": 1000,
    "snes_rtol": 1.e-10,
    "snes_atol": 1.e-10,
    "snes_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
solver_parameters_fieldsplit = {
    #'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'fieldsplit_schur_fact_type': 'full',
    "pc_fieldsplit_0_fields": "0, 2",
    "pc_fieldsplit_1_fields": "1",
    'fieldsplit_0': {
        'ksp_type': 'preonly',
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
        #'ksp_type': 'gmres',  # equationBC is nonsym
        #'pc_type': 'python',
        #'pc_python_type': 'firedrake.AssembledPC',
        #'assembled_pc_type': 'gamg',
        #'assembled_mg_levels_pc_type': 'sor',
        #'assembled_mg_levels_pc_sor_diagonal_shift': 1e-100,
        #'ksp_rtol': 1e-7,
        #'ksp_converged_reason': None,
    },
    'fieldsplit_1': {
        'ksp_type': 'preonly',
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
        #'ksp_type': 'fgmres',
        #'ksp_converged_reason': None,
        #'pc_type': 'python',
        #'pc_python_type': 'firedrake.MassInvPC',
        #'Mp_pc_type': 'ksp',
        #'Mp_ksp_ksp_type': 'cg',
        #'Mp_ksp_pc_type': 'sor',
        #'ksp_rtol': '1e-5',
        #'ksp_monitor': None,
    },
    "snes_monitor": None,
    'ksp_monitor': None,
    'ksp_view': None,
}
problem = NonlinearVariationalProblem(residual, solution, bcs=[bc_v_f_inflow, bc_v_f_zero, bc_v_f_noslip, bc_u_f_zero, bc_u_f_noslip, bc_v_s_zero, bc_u_s_zero])
solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
start = time.time()
print("num cells = ", mesh.comm.allreduce(mesh.cell_set.size), flush=True)
print("num DoFs = ", V.dim(), flush=True)
if mesh.comm.rank == 0:
    print(f"nu_ale = {float(nu_ale)}")
    print(f"mu_ale = {float(mu_ale)}")
    print(f"labmda_ale = {float(lambda_ale)}")
F_f_, J_f_, _, _, F_f_m_, J_f_m_ = compute_elast_tensors(dim, u_f, lambda_s, mu_s, u_f_m)
sigma_f_ = - p * Identity(dim) + rho_f * nu_f * 2 * sym(dot(dot(grad(v_f), inv(F_f_m_)), inv(F_f_)))

if os.path.exists(fname_checkpoint + ".h5"):
    with DumbCheckpoint(fname_checkpoint, mode=FILE_READ) as chk:
        steps, indices = chk.get_timesteps()
        t.assign(steps[-1])
        iplot = indices[-1]
        if False:
            iplot = 0
            while True:
                if abs(steps[iplot] - 3.75) < 1.e-6:
                    t.assign(steps[iplot])
                    break
                else:
                    iplot += 1
        print(f"loaded solution at t = {float(t)}")
        chk.set_timestep(float(t), iplot)
        for subsolution, subfunction_0 in zip(solution.subfunctions, solution_0.subfunctions):
            chk.load(subsolution)
            subfunction_0.assign(subsolution)
else:
    iplot = 0
    if mesh.comm.rank == 0:
        with open(fname_FD, 'w') as outfile:
             outfile.write("t val" + "\n")
        with open(fname_FL, 'w') as outfile:
             outfile.write("t val" + "\n")
        with open(fname_ux, 'w') as outfile:
             outfile.write("t val" + "\n")
        with open(fname_uy, 'w') as outfile:
             outfile.write("t val" + "\n")
if False:
    if True:  # plot mesh
        if True:
            coordV = VectorFunctionSpace(mesh_f, "P", degree)
            coords = Function(coordV).project(mesh_f.coordinates + 0*u_f_m)
            mesh = Mesh(coords)
            V = FunctionSpace(mesh, "P", degree)
            vplot = Function(V, name="mesh_f").assign(Constant(1))
            pgfplot(vplot, "mesh_f_true.dat", degree=2)
        else:
            coordV = VectorFunctionSpace(mesh_s, "P", degree)
            coords = Function(coordV).project(mesh_s.coordinates + 0*u_s_m)
            mesh = Mesh(coords)
            V = FunctionSpace(mesh, "P", degree)
            vplot = Function(V, name="mesh_s").assign(Constant(0))
            pgfplot(vplot, "mesh_s_true.dat", degree=2)
    else:  # plot fluid solution
        V = FunctionSpace(mesh_f, "P", degree)
        vplot = Function(V, name="solution_f")
        vf = solution.subfunctions[0]
        solve(inner(TrialFunction(V), TestFunction(V)) * dx_f == inner(sqrt(dot(vf, vf)), TestFunction(V)) * dx_f, vplot)
        coords = vplot.function_space().mesh().coordinates
        coords.project(mesh_f.coordinates + u_f_m + solution.subfunctions[2])
        pgfplot(vplot, "solution_f_biharmonic.dat", degree=2)
        #pgfplot(solution.subfunctions[4], "pressure.dat", degree=2)
        #pgfplot(solution.subfunctions[0], "quiver.dat", degree=0)
    raise RuntimeError("only plotted solution")
ii = 0
u_3 = Function(VectorFunctionSpace(mesh_s, "P", degree))
while float(t) < T:
    start_time = time.time()
    t.assign(float(t) + float(dt))
    ii += 1
    if mesh.comm.rank == 0:
        print(f"Computing solution at time = {float(t)} (dt = {float(dt)}, CNshift={CNshift})", flush=True)
    solver.solve()
    for subfunction, subfunction_0 in zip(solution.subfunctions, solution_0.subfunctions):
        subfunction_0.assign(subfunction)
    # Everything is now up to date.
    FD = assemble(-dot(sigma_f_, dot(dot(J_f_ * transpose(inv(F_f_)), J_f_m_ * transpose(inv(F_f_m_))), n_f))[0] * ds_f_only(subdomain_id=(label_cylinder, label_interface), degree=nquad))
    FL = assemble(-dot(sigma_f_, dot(dot(J_f_ * transpose(inv(F_f_)), J_f_m_ * transpose(inv(F_f_m_))), n_f))[1] * ds_f_only(subdomain_id=(label_cylinder, label_interface), degree=nquad))
    u_A = u_3.project(solution.subfunctions[3], solver_parameters={"ksp_rtol": 1.e-10}).at(pointA, tolerance=1.e-6)
    if mesh.comm.rank == 0:
        print(f"FD     = {FD}")
        print(f"FL     = {FL}")
        print(f"uA     = {u_A}")
        if ii % chkstride == 0:
            with open(fname_FD, 'a') as outfile:
                outfile.write(f"{float(t)} {FD}" + "\n")
            with open(fname_FL, 'a') as outfile:
                outfile.write(f"{float(t)} {FL}" + "\n")
            with open(fname_ux, 'a') as outfile:
                outfile.write(f"{float(t)} {u_A[0]}" + "\n")
            with open(fname_uy, 'a') as outfile:
                outfile.write(f"{float(t)} {u_A[1]}" + "\n")
    if ii % chkstride == 0:
        iplot += 1
        with DumbCheckpoint(fname_checkpoint, mode=FILE_UPDATE) as chk:
            chk.set_timestep(float(t), iplot)
            for subsolution in solution.subfunctions:
                chk.store(subsolution)
    gc.collect()
    end_time = time.time()
    if mesh.comm.rank == 0:
        print(f"Time per timestep: {end_time - start_time}", flush=True)
end = time.time()
print(f"Time: {end - start}")


snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:20]:
    print("rank = ", mesh.comm.rank, " : ", stat)
