import os
import time
import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import triplot, plot, quiver
from firedrake.cython import dmcommon


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
label_circle = 5
label_interface = 6


def make_mesh_netgen(h, nref, degree):
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
    circle = Circle(center=(Cx, Cy), radius=r)
    rect1 = Rectangle(pmin=(Cx, 0.19), pmax=(x5, 0.21))
    fluid_struct = rect0 - circle
    fluid = fluid_struct - rect1
    struct = fluid_struct * rect1
    geo.Add(fluid)
    geo.Add(struct)
    ngmesh = geo.GenerateMesh(maxh=h)
    mesh = Mesh(ngmesh)
    if nref > 0:
        mesh = MeshHierarchy(mesh, nref, netgen_flags={"degree": 1})[-1]
    V = FunctionSpace(mesh, "HDiv Trace", 0)
    f0 = Function(V)  # to empty labels
    f1 = Function(V)
    DirichletBC(V, 1., (4, )).apply(f1)
    f2 = Function(V)
    DirichletBC(V, 1., (2, )).apply(f2)
    f3 = Function(V)
    DirichletBC(V, 1., (1, )).apply(f3)
    f4 = Function(V)
    DirichletBC(V, 1., (3, )).apply(f4)
    f5 = Function(V)
    DirichletBC(V, 1., (5, 6, 7, 8, 12, 13 )).apply(f5)
    f6 = Function(V)
    DirichletBC(V, 1., (9, 10, 11)).apply(f6)
    return RelabeledMesh(mesh, [f0 for i in range(1, 14)] + [f1, f2, f3, f4, f5, f6], [i for i in range(1, 14)] + [1, 2, 3, 4, 5, 6])


def make_mesh(quadrilateral):
    h = 0.02
    xarray = np.concatenate([np.linspace(x0, x1, int(np.ceil((x1 - x0) / h)), endpoint=False),
                             np.linspace(x1, x2, int(np.ceil((x2 - x1) / h)), endpoint=False),
                             np.linspace(x2, x3, int(np.ceil((x3 - x2) / h)), endpoint=False),
                             np.linspace(x3, x4, int(np.ceil((x4 - x3) / h)), endpoint=False),
                             np.linspace(x4, x5, int(np.ceil((x5 - x4) / h)), endpoint=False),
                             np.linspace(x5, x6, int(np.ceil((x6 - x5) / h)) + 1, endpoint=True)])
    yarray = np.concatenate([np.linspace(y0, y1, int(np.ceil((y1 - y0) / h)), endpoint=False),
                             np.linspace(y1, y2, int(np.ceil((y2 - y1) / h)), endpoint=False),
                             np.array([y2, 0.19, 0.21]),
                             np.linspace(y3, y4, int(np.ceil((y4 - y3) / h)), endpoint=False),
                             np.linspace(y4, y5, int(np.ceil((y5 - y4) / h)) + 1, endpoint=True)])
    mesh = TensorRectangleMesh(xarray, yarray, quadrilateral=quadrilateral)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "DG", 0)
    f = Function(V).interpolate(conditional(Or(Or(x < x2, x > x3), Or(y < y2, y > y3)), 1., 0.))
    mesh = RelabeledMesh(mesh, [f], [999])
    mesh = Submesh(mesh, dmcommon.CELL_SETS_LABEL, 999, mesh.topological_dimension())  # fluid + struct
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "DG", 0)
    f_fluid = Function(V).interpolate(conditional(Or(Or(x < x3, x > x5), Or(y < 0.19, y > 0.21)), 1., 0.))
    f_struct = Function(V).interpolate(conditional(And(And(x > x3, x < x5), And(y > 0.19, y < 0.21)), 1., 0.))
    V = FunctionSpace(mesh, "HDiv Trace", 0)
    f_interface = Function(V).interpolate(conditional(Or(And(And(x > x3, x < x5), Or(And(y > 0.189, y < 0.191), And(y > 0.209, y < 0.211))), 
                                                      And(And(x > x5 - .001, x < x5 + .001), And(y > 0.19, y < 0.21))), 1., 0.))
    return RelabeledMesh(mesh, [f_fluid, f_struct, f_interface], [label_fluid, label_struct, label_interface])


def _finalise_mesh(mesh, degree):
    V = VectorFunctionSpace(mesh, "CG", degree)
    f = Function(V).interpolate(mesh.coordinates)
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
    return Mesh(f, name=mesh.name)



def _elevate_degree(mesh, degree):
    V = VectorFunctionSpace(mesh, "CG", degree)
    f = Function(V).interpolate(mesh.coordinates)
    bc = DirichletBC(V, 0., label_circle)
    r_ = np.sqrt((f.dat.data_with_halos[bc.nodes, 0] - Cx) ** 2 + ((f.dat.data_with_halos[bc.nodes, 1] - Cy) ** 2))
    f.dat.data_with_halos[bc.nodes, 0] = (f.dat.data_with_halos[bc.nodes, 0] - Cx) * (r / r_) + Cy
    f.dat.data_with_halos[bc.nodes, 1] = (f.dat.data_with_halos[bc.nodes, 1] - Cy) * (r / r_) + Cy
    return Mesh(f)


dim = 2
use_netgen = True
quadrilateral = False
degree = 4  # 2 - 4
if use_netgen:
    nref = 2 #  # 2 - 5 tested for CSM 1 and 2
    mesh  = make_mesh_netgen(0.12, nref, degree)
    mesh = _elevate_degree(mesh, degree)
    mesh_f = Submesh(mesh, dmcommon.CELL_SETS_LABEL, label_fluid, mesh.topological_dimension())
    mesh_f = _elevate_degree(mesh_f, degree)
    mesh_s = Submesh(mesh, dmcommon.CELL_SETS_LABEL, label_struct, mesh.topological_dimension())
    mesh_s = _elevate_degree(mesh_s, degree)
else:
    nref = 0
    mesh = make_mesh(quadrilateral)
    if nref > 0:
        mesh = MeshHierarchy(mesh, nref)[-1]
    mesh_f = Submesh(mesh, dmcommon.CELL_SETS_LABEL, label_fluid, mesh.topological_dimension(), name="mesh_fluid")
    mesh_s = Submesh(mesh, dmcommon.CELL_SETS_LABEL, label_struct, mesh.topological_dimension(), name="mesh_struct")
    mesh = _finalise_mesh(mesh, degree)
    mesh_f = _finalise_mesh(mesh_f, degree)
    mesh_s = _finalise_mesh(mesh_s, degree)

"""
#mesh.topology_dm.viewFromOptions("-dm_view")
v = assemble(Constant(1.0, domain=mesh) * ds(label_circle))
print(v - 2 * pi * r)
print(assemble(x * dx(label_struct)))
print((0.6 - 0.248989794855664) * (0.6 + 0.248989794855664) /2. * 0.02)
print(assemble(Constant(1) * dx(domain=mesh, subdomain_id=label_fluid)) + assemble(Constant(1) * dx(domain=mesh, subdomain_id=label_struct)))
print(assemble(Constant(1) * dx(domain=mesh)))
print(L * H - pi * r ** 2)
"""

x, y = SpatialCoordinate(mesh)
x_f, y_f = SpatialCoordinate(mesh_f)
x_s, y_s = SpatialCoordinate(mesh_s)
n_f = FacetNormal(mesh_f)
n_s = FacetNormal(mesh_s)
dx = Measure("dx", domain=mesh)
dx_f = Measure("dx", domain=mesh_f)
dx_s = Measure("dx", domain=mesh_s)
ds = Measure("ds", domain=mesh)
ds_f = Measure("ds", domain=mesh_f)
ds_s = Measure("ds", domain=mesh_s)
dS = Measure("dS", domain=mesh)
dS_f = Measure("dS", domain=mesh_f)
dS_s = Measure("dS", domain=mesh_s)

if mesh.comm.size == 1:
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

case = "FSI3_2"

if case in ["CFD1", "CFD2", "CFD3"]:
    T = 20 # 10.0 # 12.0
    dt = Constant(0.0005)  #0.001
    dt_plot = 0.01
    ntimesteps = int(T / float(dt))
    t = Constant(0.0)
    CNshift = 2
    if case == "CFD1":
        rho_f = 1.e+3
        nu_f = 1.e-3
        Ubar = 0.2
        # Re = 20.
    elif case == "CFD2":
        rho_f = 1.e+3
        nu_f = 1.e-3
        Ubar = 1.
        # Re = 100.
    elif case == "CFD3":
        rho_f = 1.e+3
        nu_f = 1.e-3
        Ubar = 2.
        # Re = 200.
    else:
        raise ValueError
    V_0 = VectorFunctionSpace(mesh_f, "P", degree)
    #V_1 = FunctionSpace(mesh_f, "P", degree - 2)
    V_1 = FunctionSpace(mesh_f, "P", degree - 1)
    V = V_0 * V_1
    solution = Function(V)
    solution_0 = Function(V)
    v_f, p_f = split(solution)
    v_f_0, p_f_0 = split(solution_0)
    dv_f, dp_f = split(TestFunction(V))
    residual = (inner(rho_f * (v_f - v_f_0) / dt, dv_f) +
                inner(rho_f * (dot(grad(v_f), v_f) + dot(grad(v_f_0), v_f_0)) / 2, dv_f) +
                inner(rho_f * nu_f * (2 * sym(grad(v_f)) + 2 * sym(grad(v_f_0))) / 2, grad(dv_f)) -
                inner((p_f + p_f_0) / 2, div(dv_f)) +
                inner(div(v_f), dp_f)) * dx_f
    v_f_left = 1.5 * Ubar * y_f * (H - y_f) / ((H / 2) ** 2) * conditional(t < 2.0, (1 - cos(pi / 2 * t)) / 2., 1.)
    bc_inflow = DirichletBC(V.sub(0), as_vector([v_f_left, 0.]), (label_left, ))
    bc_noslip = DirichletBC(V.sub(0), Constant((0, 0)), (label_bottom, label_top, label_circle, label_interface))
    solver_parameters = {
        "mat_type": "aij",
        "snes_monitor": None,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    solver_parameters = {
        #'mat_type': 'matfree',
        'pc_type': 'fieldsplit',
        'ksp_type': 'preonly',
        'pc_fieldsplit_type': 'schur',
        'fieldsplit_schur_fact_type': 'full',
        'fieldsplit_0': {
            #'ksp_type': 'cg',
            'ksp_type': 'gmres',  # equationBC is nonsym
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'gamg',
            'assembled_mg_levels_pc_type': 'jacobi',
            'assembled_mg_levels_pc_sor_diagonal_shift': 1e-100,  # See https://gitlab.com/petsc/petsc/-/issues/1221
            'ksp_rtol': 1e-13,
            'ksp_converged_reason': None,
            'ksp_monitor': None,
        },
        'fieldsplit_1': {
            'ksp_type': 'fgmres',
            'ksp_converged_reason': None,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.MassInvPC',
            'Mp_pc_type': 'ksp',
            #'Mp_ksp_ksp_type': 'cg',
            'Mp_ksp_ksp_type': 'gmres',
            'Mp_ksp_pc_type': 'sor',
            'ksp_rtol': '1e-10',
            'ksp_monitor': None,
        },
        "snes_monitor": None,
        "ksp_monitor": None,
    }
    problem = NonlinearVariationalProblem(residual, solution, bcs=[bc_inflow, bc_noslip])
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
    for itimestep in range(ntimesteps):
        if mesh.comm.rank == 0:
            print(f"{itimestep} / {ntimesteps}", flush=True)
        t.assign((itimestep + 1) * float(dt))
        solver.solve()
        for subfunction, subfunction_0 in zip(solution.subfunctions, solution_0.subfunctions):
            subfunction_0.assign(subfunction)
        FD = assemble((- p_f * n_f + rho_f * nu_f * dot(2 * sym(grad(v_f)), n_f))[0] * ds_f((label_circle, label_interface)))
        FL = assemble((- p_f * n_f + rho_f * nu_f * dot(2 * sym(grad(v_f)), n_f))[1] * ds_f((label_circle, label_interface)))
        if mesh.comm.rank == 0:
            print(f"FD = {FD}")
            print(f"FL = {FL}")
    print("num cells = ", mesh_f.comm.allreduce(mesh_f.cell_set.size))
elif case in ["CSM1", "CSM2", "CSM3"]:
    T = 20 # 10.0 # 12.0
    dt = Constant(0.0005)  #0.001
    dt_plot = 0.01
    ntimesteps = int(T / float(dt))
    t = Constant(0.0)
    CNshift = 2
    if case == "CSM1":
        rho_s = 1.e+3
        nu_s = 0.4
        E_s = 1.4 * 1.e+6
    elif case == "CSM2":
        rho_s = 1.e+3
        nu_s = 0.4
        E_s = 5.6 * 1.e+6
    elif case == "CSM3":
        rho_s = 1.e+3
        nu_s = 0.4
        E_s = 1.4 * 1.e+6
    else:
        raise ValueError
    g_s_float = 2.0
    g_s = Constant(0.)
    lambda_s = nu_s * E_s / (1 + nu_s) / (1 - 2 * nu_s)
    mu_s = E_s / 2 / (1 + nu_s)
    if case in ["CSM1", "CSM2"]:
        V = VectorFunctionSpace(mesh_s, "P", degree)
        u_s = Function(V)
        du_s = TestFunction(V)
        F = Identity(dim) + grad(u_s)
        E = 1. / 2. * (dot(transpose(F), F) - Identity(dim))
        S = lambda_s * tr(E) * Identity(dim) + 2.0 * mu_s * E
        residual = inner(dot(F, S), grad(du_s)) * dx_s - \
                   rho_s * inner(as_vector([0., - g_s]), du_s) * dx_s
        bc = DirichletBC(V, as_vector([0., 0.]), (label_circle, ))
        solver_parameters = {
            "mat_type": "aij",
            "snes_monitor": None,
            "ksp_monitor": None,
            #"ksp_view": None,
            "ksp_type": "gmres",
            "pc_type": "gamg",
            "mg_levels_pc_type": "sor",
            'mg_levels_ksp_max_it': 3,
            #"pc_type": "lu",
            #"pc_factor_mat_solver_type": "mumps"
        }
        near_nullspace = VectorSpaceBasis(vecs=[assemble(Function(V).interpolate(rigid_body_mode)) \
                                                for rigid_body_mode in [Constant((1, 0)), Constant((0, 1)), as_vector([y_s, -x_s])]])
        near_nullspace.orthonormalize()
        problem = NonlinearVariationalProblem(residual, u_s, bcs=[bc])
        solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters, near_nullspace=near_nullspace)
        # Use relaxation method.
        nsteps = 10
        for g_s_temp in [g_s_float * (i + 1) / nsteps for i in range(nsteps)]:
            g_s.assign(g_s_temp)
            solver.solve()
        # (degree, nref) = (2-4, 2-4) with mumps work. .1 * gs
        # (4, 5) has 280,962 DoFs.
        # u_s.at(pointA) = [-0.00722496 -0.06629327]
        print(V.dim())
        print(u_s.at(pointA))
        #assert np.allclose(u_s.at(pointA), [-0.007187, -0.06610], rtol=1e-03)
    else:  # CSM3
        g_s.assign(g_s_float)
        V0 = VectorFunctionSpace(mesh_s, "P", degree)
        V = V0 * V0
        solution = Function(V)
        solution_0 = Function(V)
        v_s, u_s = split(solution)
        v_s_0, u_s_0 = split(solution_0)
        dv_s, du_s = split(TestFunction(V))
        F = Identity(dim) + grad(u_s)
        E = 1. / 2. * (dot(transpose(F), F) - Identity(dim))
        S = lambda_s * tr(E) * Identity(dim) + 2.0 * mu_s * E
        F_0 = Identity(dim) + grad(u_s_0)
        E_0 = 1. / 2. * (dot(transpose(F_0), F_0) - Identity(dim))
        S_0 = lambda_s * tr(E_0) * Identity(dim) + 2.0 * mu_s * E_0
        residual = inner((u_s - u_s_0) / dt, dv_s) * dx_s - \
                   inner((v_s + v_s_0) / 2, dv_s) * dx_s + \
                   inner(rho_s * (v_s - v_s_0) / dt, du_s) * dx_s + \
                   inner((dot(F, S) + dot(F_0, S_0)) / 2, grad(du_s)) * dx_s - \
                   inner(rho_s * as_vector([0., - g_s]), du_s) * dx_s
        bc_v = DirichletBC(V.sub(0), as_vector([0., 0.]), (label_circle, ))
        bc_u = DirichletBC(V.sub(1), as_vector([0., 0.]), (label_circle, ))
        solver_parameters = {
            "mat_type": "aij",
            #"snes_monitor": None,
            #"ksp_monitor": None,
            #"ksp_view": None,
            "ksp_type": "gmres",
            #"pc_type": "gamg",
            #"mg_levels_pc_type": "sor",
            #'mg_levels_ksp_max_it': 3,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }
        problem = NonlinearVariationalProblem(residual, solution, bcs=[bc_v, bc_u])
        solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
        for itimestep in range(ntimesteps):
            if mesh.comm.rank == 0:
                print(f"{itimestep} / {ntimesteps}", flush=True)
            t.assign((itimestep + 1) * float(dt))
            solver.solve()
            for subfunction, subfunction_0 in zip(solution.subfunctions, solution_0.subfunctions):
                subfunction_0.assign(subfunction)
            u_A = solution.subfunctions[1].at(pointA)
            if mesh.comm.rank == 0:
                print(u_A)
elif case in ["FSI1", "FSI2", "FSI3"]:
    T = 20 # 10.0 # 12.0
    dt = Constant(0.002)  #0.001
    dt_plot = 0.01
    t = Constant(0.0)
    CNshift = 10
    elast = True
    linear_elast = True
    fname_checkpoint = f"dumbdata/fsi3_0_Q4_Q3_nref{nref}_0.002_shift{CNshift}_{elast}_{linear_elast}"
    fname_FD = f"time_series_0_FD_Q4_Q3_nref{nref}_0.002_shift{CNshift}_{elast}_{linear_elast}.dat"
    fname_FL = f"time_series_0_FL_Q4_Q3_nref{nref}_0.002_shift{CNshift}_{elast}_{linear_elast}.dat"
    if case == "FSI1":
        rho_s = 1.e+3
        nu_s = 0.4
        mu_s = 0.5 * 1.e+6
        rho_f = 1.e+3
        nu_f = 1.e-3
        Ubar = 0.2
        # Re = 20.
    elif case == "FSI2":
        rho_s = 10. * 1.e+3
        nu_s = 0.4
        mu_s = 0.5 * 1.e+6
        rho_f = 1.e+3
        nu_f = 1.e-3
        Ubar = 1.0
        # Re = 100.
    elif case == "FSI3":
        rho_s = 1. * 1.e+3
        nu_s = 0.4
        mu_s = 2.0 * 1.e+6
        rho_f = 1.e+3
        nu_f = 1.e-3
        Ubar = 2.0
        # Re = 200.
    else:
        raise ValueError
    #g_s = Constant(2.0)
    g_s = Constant(0.0)
    E_s = mu_s * 2 * (1 + nu_s)
    lambda_s = nu_s * E_s / (1 + nu_s) / (1 - 2 * nu_s)
    V_0 = VectorFunctionSpace(mesh, "CG", degree)
    V_1 = FunctionSpace(mesh_f, "CG", degree - 1)
    V = V_0 * V_0 * V_1
    solution = Function(V)
    solution_0 = Function(V)
    v, u, p = split(solution)
    v_0, u_0, p_0 = split(solution_0)
    dv, du, dp = split(TestFunction(V))
    for subf, name in zip(solution.subfunctions, ["v", "u", "p"]):
        subf.rename(name)
    def compute_elast_tensors(dim, u, lambda_s, mu_s):
        F = Identity(dim) + grad(u)
        J = det(F)
        E = 1. / 2. * (dot(transpose(F), F) - Identity(dim))
        S = lambda_s * tr(E) * Identity(dim) + 2.0 * mu_s * E
        return F, J, E, S
    if True:
        theta_p = Constant(1. / 2. + CNshift * float(dt))
        theta_m = Constant(1. / 2. - CNshift * float(dt))
        v_dot = (v - v_0) / dt
        u_dot = (u - u_0) / dt
        def _fluid_elast_lin(v_f, u_f, p):
            F_f, J_f, E_f, S_f = compute_elast_tensors(dim, u_f, lambda_s, mu_s)
            epsilon = sym(grad(u_f))
            sigma = 2 * mu_s * epsilon + lambda_s * tr(epsilon) * Identity(dim)
            return (inner(rho_f * J_f * v_dot, dv) +
                    inner(rho_f * J_f * dot(dot(grad(v_f), inv(F_f)), v_f - u_dot), dv) +
                    inner(rho_f * J_f * nu_f * 2 * sym(dot(grad(v_f), inv(F_f))), dot(grad(dv), inv(F_f))) -
                    J_f * inner(p, tr(dot(grad(dv), inv(F_f)))) +
                    J_f * inner(tr(dot(grad(v_f), inv(F_f))), dp) +
                    inner(sigma, grad(du))) * dx_f
        def _struct(v_s, u_s):
            F_s, J_s, E_s, S_s = compute_elast_tensors(dim, u_s, lambda_s, mu_s)
            return (inner(rho_s * J_s * v_dot, dv) +
                    inner(dot(F_s, S_s), grad(dv)) -
                    inner(rho_s * J_s * as_vector([0., - g_s]), dv) +
                    inner(J_s * (u_dot - v_s), du)) * dx_s
        residual_f = theta_p * _fluid_elast_lin(v, u, p) + \
                     theta_m * _fluid_elast_lin(v_0, u_0, p_0)
        residual_s = theta_p * _struct(v, u) + \
                     theta_m * _struct(v_0, u_0)
    residual = residual_f + residual_s
    def v_f_left(t_):
        return 1.5 * Ubar * y * (H - y) / ((H / 2) ** 2) * conditional(t_ < 2.0 + dt / 10., (1 - cos(pi / 2 * t_)) / 2., 1.)
    bc_inflow = DirichletBC(V.sub(0), as_vector([theta_p * v_f_left(t - dt + theta_p * dt) +
                                                 theta_m * v_f_left(t - dt + theta_m * dt), 0.]), (label_left, ))
    bc_noslip_v = DirichletBC(V.sub(0), Constant((0, 0)), (label_bottom, label_top, label_circle))
    bc_noslip_u = DirichletBC(V.sub(1), Constant((0, 0)), (label_left, label_right, label_bottom, label_top, label_circle))
    bbc = DirichletBC(V.sub(1), Constant((0, 0)), ((label_circle, label_interface), ))
    bc_noslip_u_int = EquationBC(inner((u('+') - u_0('+')) / dt - v('+'), du('+')) * ds_f(label_interface) == 0, solution, label_interface, bcs=[bbc], V=V.sub(1))
    solver_parameters = {
        "mat_type": "aij",
        "snes_rtol": 1.e-10,
        "snes_atol": 1.e-10,
        "snes_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
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
    #problem = NonlinearVariationalProblem(residual, solution, bcs=[bc_inflow, bc_noslip_v, bc_noslip_u, bc_noslip_v_int, bc_noslip_u_int])
    problem = NonlinearVariationalProblem(residual, solution, bcs=[bc_inflow, bc_noslip_v, bc_noslip_u, bc_noslip_u_int])
    #problem = NonlinearVariationalProblem(residual, solution, bcs=[bc_inflow, bc_noslip_v, bc_noslip_u])
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
    start = time.time()
    nsample = int(T / dt_plot)
    sample_t = np.arange(0.0, T, dt_plot) + dt_plot
    sample_FD = np.empty_like(sample_t)
    sample_FL = np.empty_like(sample_t)
    print("num cells = ", mesh.comm.allreduce(mesh.cell_set.size), flush=True)
    print("num DoFs = ", V.dim(), flush=True)
    v_split = Function(VectorFunctionSpace(mesh_f, "P", degree)).interpolate(solution.subfunctions[0])
    u_split = Function(VectorFunctionSpace(mesh_f, "P", degree)).interpolate(solution.subfunctions[1])
    F_f_, J_f_, _, _ = compute_elast_tensors(dim, u_split, lambda_s, mu_s)
    sigma_f_ = - p * Identity(dim) + rho_f * nu_f * 2 * sym(dot(grad(v_split), inv(F_f_)))
    if os.path.exists(fname_checkpoint + ".h5"):
        with DumbCheckpoint(fname_checkpoint, mode=FILE_READ) as chk:
            steps, indices = chk.get_timesteps()
            t.assign(steps[-1])
            iplot = indices[-1]
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
    ii = 0
    while float(t) < T:
        t.assign(float(t) + float(dt))
        ii += 1
        if mesh.comm.rank == 0:
            print(f"Computing solution at time = {float(t)} (dt = {float(dt)}, CNshift={CNshift})", flush=True)
        solver.solve()
        for subfunction, subfunction_0 in zip(solution.subfunctions, solution_0.subfunctions):
            subfunction_0.assign(subfunction)
        # Everything is now up to date.
        FD = assemble(dot(sigma_f_, dot(J_f_ * transpose(inv(F_f_)), n_f))[0] * ds_f((label_circle, label_interface)))
        FL = assemble(dot(sigma_f_, dot(J_f_ * transpose(inv(F_f_)), n_f))[1] * ds_f((label_circle, label_interface)))
        u_A = solution.subfunctions[1].at(pointA, tolerance=1.e-6)
        if mesh.comm.rank == 0:
            print(f"FD     = {FD}")
            print(f"FL     = {FL}")
            print(f"uA     = {u_A}")
            if ii % 10 == 0:
                with open(fname_FD, 'a') as outfile:
                    outfile.write(f"{float(t)} {FD}" + "\n")
                with open(fname_FL, 'a') as outfile:
                    outfile.write(f"{float(t)} {FL}" + "\n")
                    #sample_FD[itimestep // (ntimesteps // nsample)] = FD
                    #sample_FL[itimestep // (ntimesteps // nsample)] = FL
                    #np.savetxt(outfile, np.concatenate([sample_t.reshape(-1, 1), sample_FD.reshape(-1, 1)], axis=1))
        if ii % 10 == 0:
            iplot += 1
            with DumbCheckpoint(fname_checkpoint, mode=FILE_UPDATE) as chk:
                chk.set_timestep(float(t), iplot)
                for subsolution in solution.subfunctions:
                    chk.store(subsolution)

    end = time.time()
    print(f"Time: {end - start}")
elif case in ["FSI1_2", "FSI2_2", "FSI3_2"]:
    T = 20 # 10.0 # 12.0
    dt = Constant(0.002)  #0.001
    dt_plot = 0.01
    t = Constant(0.0)
    CNshift = 10
    elast = True
    linear_elast = True
    if use_netgen:
        fname_checkpoint = f"dumbdata/fsi3_P4_P2_nref{nref}_0.002_shift{CNshift}_{elast}_{linear_elast}_netgen_new"
        fname_FD = f"time_series_FD_P4_P2_nref{nref}_0.002_shift{CNshift}_{elast}_{linear_elast}_netgen_new.dat"
        fname_FL = f"time_series_FL_P4_P2_nref{nref}_0.002_shift{CNshift}_{elast}_{linear_elast}_netgen_new.dat"
    else:
        if quadrilateral:
            fname_checkpoint = f"dumbdata/fsi3_Q4_Q3_nref{nref}_0.001_shift{CNshift}_{elast}_{linear_elast}"
            fname_FD = f"time_series_FD_Q4_Q3_nref{nref}_0.001_shift{CNshift}_{elast}_{linear_elast}.dat"
            fname_FL = f"time_series_FL_Q4_Q3_nref{nref}_0.001_shift{CNshift}_{elast}_{linear_elast}.dat"
        else:
            fname_checkpoint = f"dumbdata/fsi3_P4_P2_nref{nref}_0.002_shift{CNshift}_{elast}_{linear_elast}_ALEparamtest1"
            fname_FD = f"time_series_FD_P4_P2_nref{nref}_0.002_shift{CNshift}_{elast}_{linear_elast}_ALEparamtest1.dat"
            fname_FL = f"time_series_FL_P4_P2_nref{nref}_0.002_shift{CNshift}_{elast}_{linear_elast}_ALEparamtest1.dat"
    if case == "FSI1_2":
        rho_s = 1.e+3
        nu_s = 0.4
        mu_s = 0.5 * 1.e+6
        rho_f = 1.e+3
        nu_f = 1.e-3
        Ubar = 0.2
        # Re = 20.
    elif case == "FSI2_2":
        rho_s = 10. * 1.e+3
        nu_s = 0.4
        mu_s = 0.5 * 1.e+6
        rho_f = 1.e+3
        nu_f = 1.e-3
        Ubar = 1.0
        # Re = 100.
    elif case == "FSI3_2":
        rho_s = 1. * 1.e+3
        nu_s = 0.4
        mu_s = 2.0 * 1.e+6
        rho_f = 1.e+3
        nu_f = 1.e-3
        Ubar = 2.0
        # Re = 200.
    else:
        raise ValueError
    g_s = Constant(0.0)
    E_s = mu_s * 2 * (1 + nu_s)
    lambda_s = nu_s * E_s / (1 + nu_s) / (1 - 2 * nu_s)
    # ALE constants
    nu_ale = Constant(float(nu_s))
    mu_ale = Constant(float(mu_s))
    E_ale = mu_ale * 2 * (1 + nu_ale)
    lambda_ale = nu_ale * E_ale / (1 + nu_ale) / (1 - 2 * nu_ale)
    if use_netgen or not quadrilateral:
        V_0 = VectorFunctionSpace(mesh_f, "P", degree)
        V_1 = VectorFunctionSpace(mesh_s, "P", degree)
        V_2 = FunctionSpace(mesh_f, "P", degree - 2)
    else:
        V_0 = VectorFunctionSpace(mesh_f, "Q", degree)
        V_1 = VectorFunctionSpace(mesh_s, "Q", degree)
        V_2 = FunctionSpace(mesh_f, "Q", degree - 1)
    V = V_0 * V_1 * V_0 * V_1 * V_2
    solution = Function(V)
    solution_0 = Function(V)
    v_f, v_s, u_f, u_s, p = split(solution)
    v_f_0, v_s_0, u_f_0, u_s_0, p_0 = split(solution_0)
    dv_f, dv_s, du_f, du_s, dp = split(TestFunction(V))
    for subf, name in zip(solution.subfunctions, ["v_f", "v_s", "u_f", "u_s", "p"]):
        subf.rename(name)
    def compute_elast_tensors(dim, u, lambda_s, mu_s):
        F = Identity(dim) + grad(u)
        J = det(F)
        E = 1. / 2. * (dot(transpose(F), F) - Identity(dim))
        S = lambda_s * tr(E) * Identity(dim) + 2.0 * mu_s * E
        return F, J, E, S
    if False:  # implicit midpoint
        theta_p = Constant(1. / 2.)
        theta_m = Constant(1. / 2.)
        F_f, J_f, E_f, S_f = compute_elast_tensors(dim, (u_f + u_f_0) / 2, lambda_s, mu_s)
        F_s, J_s, E_s, S_s = compute_elast_tensors(dim, (u_s + u_s_0) / 2, lambda_s, mu_s)
        v_f_mid = (v_f + v_f_0) / 2
        v_s_mid = (v_s + v_s_0) / 2
        u_f_mid = (u_f + u_f_0) / 2
        p_mid = (p + p_0) / 2
        residual_f = (
            inner(rho_f * J_f * (v_f - v_f_0) / dt, dv_f) +
            inner(rho_f * J_f * dot(dot(grad(v_f_mid), inv(F_f)), v_f_mid - (u_f - u_f_0) / dt), dv_f) +
            inner(rho_f * J_f * nu_f * 2 * sym(dot(grad(v_f_mid), inv(F_f))), dot(grad(dv_f), inv(F_f))) -
            J_f * inner(p_mid, tr(dot(grad(dv_f), inv(F_f)))) +
            J_f * inner(tr(dot(grad(v_f_mid), inv(F_f))), dp) +
            J_f * inner(dot(grad(u_f_mid), inv(F_f)), dot(grad(du_f), inv(F_f)))
        ) * dx_f
        residual_s = (
            inner(rho_s * J_s * (v_s - v_s_0) / dt, dv_s) +
            inner(dot(F_s, S_s), grad(dv_s)) -
            inner(rho_s * J_s * as_vector([0., - g_s]), dv_s) +
            inner(J_s * ((u_s - u_s_0) / dt - v_s_mid), du_s)
        ) * dx_s + \
        inner(dot(- p_mid * Identity(dim) + rho_f * nu_f * 2 * sym(dot(grad(v_f_mid), inv(F_f))), dot(J_f * transpose(inv(F_f)), n_f)), dv_s('|')) * ds_s(label_interface)
        #inner(dot(- p('|') * Identity(dim) + rho_f * nu_f * 2 * sym(dot(grad(v_f('|')), inv(F_f))), dot(J_f * transpose(inv(F_f)), n_f)), dv_s('|')) * ds_s(label_interface)
    else:  # CN
        theta_p = Constant(1. / 2. + CNshift * float(dt))
        theta_m = Constant(1. / 2. - CNshift * float(dt))
        #theta_p = Constant(1.)
        #theta_m = Constant(0.)
        v_f_dot = (v_f - v_f_0) / dt
        u_f_dot = (u_f - u_f_0) / dt
        v_s_dot = (v_s - v_s_0) / dt
        u_s_dot = (u_s - u_s_0) / dt
        def _fluid_laplace(v_f, u_f, p):
            F_f, J_f, E_f, S_f = compute_elast_tensors(dim, u_f, lambda_s, mu_s)
            return (inner(rho_f * J_f * v_f_dot, dv_f) +
                    inner(rho_f * J_f * dot(dot(grad(v_f), inv(F_f)), v_f - u_f_dot), dv_f) +
                    inner(rho_f * J_f * nu_f * 2 * sym(dot(grad(v_f), inv(F_f))), dot(grad(dv_f), inv(F_f))) -
                    J_f * inner(p, tr(dot(grad(dv_f), inv(F_f)))) +
                    J_f * inner(tr(dot(grad(v_f), inv(F_f))), dp) +
                    J_f * inner(dot(grad(u_f), inv(F_f)), dot(grad(du_f), inv(F_f)))) * dx_f
        def _fluid_elast(v_f, u_f, p):
            F_f, J_f, E_f, S_f = compute_elast_tensors(dim, u_f, lambda_ale, mu_ale)
            return (inner(rho_f * J_f * v_f_dot, dv_f) +
                    inner(rho_f * J_f * dot(dot(grad(v_f), inv(F_f)), v_f - u_f_dot), dv_f) +
                    inner(rho_f * J_f * nu_f * 2 * sym(dot(grad(v_f), inv(F_f))), dot(grad(dv_f), inv(F_f))) -
                    J_f * inner(p, tr(dot(grad(dv_f), inv(F_f)))) +
                    J_f * inner(tr(dot(grad(v_f), inv(F_f))), dp) +
                    inner(dot(F_f, S_f), grad(du_f))) * dx_f
        def _fluid_elast_lin(v_f, u_f, p):
            F_f, J_f, E_f, S_f = compute_elast_tensors(dim, u_f, lambda_ale, mu_ale)
            epsilon = sym(grad(u_f))
            sigma = 2 * mu_ale * epsilon + lambda_ale * tr(epsilon) * Identity(dim)
            return (inner(rho_f * J_f * v_f_dot, dv_f) +
                    inner(rho_f * J_f * dot(dot(grad(v_f), inv(F_f)), v_f - u_f_dot), dv_f) +
                    inner(rho_f * J_f * nu_f * 2 * sym(dot(grad(v_f), inv(F_f))), dot(grad(dv_f), inv(F_f))) -
                    J_f * inner(p, tr(dot(grad(dv_f), inv(F_f)))) +
                    J_f * inner(tr(dot(grad(v_f), inv(F_f))), dp) +
                    inner(sigma, grad(du_f))) * dx_f
        def _struct(v_f, u_f, p, v_s, u_s):
            F_f, J_f, E_f, S_f = compute_elast_tensors(dim, u_f, lambda_s, mu_s)
            F_s, J_s, E_s, S_s = compute_elast_tensors(dim, u_s, lambda_s, mu_s)
            return (inner(rho_s * J_s * v_s_dot, dv_s) +
                    inner(dot(F_s, S_s), grad(dv_s)) -
                    inner(rho_s * J_s * as_vector([0., - g_s]), dv_s) +
                    inner(J_s * (u_s_dot - v_s), du_s)) * dx_s + \
                   inner(dot(- p('|') * Identity(dim) + rho_f * nu_f * 2 * sym(dot(grad(v_f('|')), inv(F_f))), dot(J_f * transpose(inv(F_f)), n_f)), dv_s('|')) * ds_s(label_interface)
        if elast:
            if linear_elast:
                residual_f = theta_p * _fluid_elast_lin(v_f, u_f, p) + \
                             theta_m * _fluid_elast_lin(v_f_0, u_f_0, p_0)
            else:
                residual_f = theta_p * _fluid_elast(v_f, u_f, p) + \
                             theta_m * _fluid_elast(v_f_0, u_f_0, p_0)
        else:
            residual_f = theta_p * _fluid_laplace(v_f, u_f, p) + \
                         theta_m * _fluid_laplace(v_f_0, u_f_0, p_0)
        residual_s = theta_p * _struct(v_f, u_f, p, v_s, u_s) + \
                     theta_m * _struct(v_f_0, u_f_0, p_0, v_s_0, u_s_0)
        """
        F_f, J_f, E_f, S_f = compute_elast_tensors(dim, u_f, lambda_s, mu_s)
        F_s, J_s, E_s, S_s = compute_elast_tensors(dim, u_s, lambda_s, mu_s)
        v_f_mid = v_f
        v_s_mid = v_s
        u_f_mid = u_f
        p_mid = p
        residual_f = (
            inner(rho_f * J_f * (v_f - v_f_0) / dt, dv_f) +
            inner(rho_f * J_f * dot(dot(grad(v_f_mid), inv(F_f)), v_f_mid - (u_f - u_f_0) / dt), dv_f) +
            inner(rho_f * J_f * nu_f * 2 * sym(dot(grad(v_f_mid), inv(F_f))), dot(grad(dv_f), inv(F_f))) -
            J_f * inner(p_mid, tr(dot(grad(dv_f), inv(F_f)))) +
            J_f * inner(tr(dot(grad(v_f_mid), inv(F_f))), dp) +
            J_f * inner(dot(grad(u_f_mid), inv(F_f)), dot(grad(du_f), inv(F_f)))
        ) * dx_f
        residual_s = (
            inner(rho_s * J_s * (v_s - v_s_0) / dt, dv_s) +
            inner(dot(F_s, S_s), grad(dv_s)) -
            inner(rho_s * J_s * as_vector([0., - g_s]), dv_s) +
            inner(J_s * ((u_s - u_s_0) / dt - v_s_mid), du_s)
        ) * dx_s + \
        inner(dot(- p('|') * Identity(dim) + rho_f * nu_f * 2 * sym(dot(grad(v_f('|')), inv(F_f))), dot(J_f * transpose(inv(F_f)), n_f)), dv_s('|')) * ds_s(label_interface)
        #inner(dot(- p_mid * Identity(dim) + rho_f * nu_f * 2 * sym(dot(grad(v_f_mid), inv(F_f))), dot(J_f * transpose(inv(F_f)), n_f)), dv_s('|')) * ds_s(label_interface)
        """
    residual = residual_f + residual_s
    def v_f_left(t_):
        return 1.5 * Ubar * y_f * (H - y_f) / ((H / 2) ** 2) * conditional(t_ < 2.0 + dt / 10., (1 - cos(pi / 2 * t_)) / 2., 1.)
    bc_v_f_inflow = DirichletBC(V.sub(0), as_vector([theta_p * v_f_left(t - dt + theta_p * dt) +
                                                     theta_m * v_f_left(t - dt + theta_m * dt), 0.]), (label_left, ))
    bc_v_f_zero = DirichletBC(V.sub(0), Constant((0, 0)), (label_bottom, label_top, label_circle))
    bbc_v_f_noslip = DirichletBC(V.sub(0), Constant((0, 0)), ((label_circle, label_interface), ))
    bc_v_f_noslip = EquationBC(inner(v_f - v_s, dv_f) * ds_f(label_interface) == 0, solution, label_interface, bcs=[bbc_v_f_noslip], V=V.sub(0))
    bc_u_f_zero = DirichletBC(V.sub(2), Constant((0, 0)), (label_left, label_right, label_bottom, label_top, label_circle))
    bbc_u_f_noslip = DirichletBC(V.sub(2), Constant((0, 0)), ((label_circle, label_interface), ))
    bc_u_f_noslip = EquationBC(inner(u_f - u_s, du_f) * ds_f(label_interface) == 0, solution, label_interface, bcs=[bbc_u_f_noslip], V=V.sub(2))
    bc_v_s_zero = DirichletBC(V.sub(1), Constant((0, 0)), (label_circle, ))
    bc_u_s_zero = DirichletBC(V.sub(3), Constant((0, 0)), (label_circle, ))
    solver_parameters = {
        "mat_type": "aij",
        "snes_mat_it": 1000,
        "snes_rtol": 1.e-10,
        "snes_atol": 1.e-10,
        "snes_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
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
    nsample = int(T / dt_plot)
    sample_t = np.arange(0.0, T, dt_plot) + dt_plot
    sample_FD = np.empty_like(sample_t)
    sample_FL = np.empty_like(sample_t)
    print("num cells = ", mesh.comm.allreduce(mesh.cell_set.size), flush=True)
    print("num DoFs = ", V.dim(), flush=True)
    if mesh.comm.rank == 0:
        print(f"nu_ale = {float(nu_ale)}")
        print(f"mu_ale = {float(mu_ale)}")
        print(f"labmda_ale = {float(lambda_ale)}")
    #v_f_ = solution.subfunctions[0]
    F_f_, J_f_, _, _ = compute_elast_tensors(dim, u_f, lambda_s, mu_s)
    sigma_f_ = - p * Identity(dim) + rho_f * nu_f * 2 * sym(dot(grad(v_f), inv(F_f_)))
    if os.path.exists(fname_checkpoint + ".h5"):
        with DumbCheckpoint(fname_checkpoint, mode=FILE_READ) as chk:
            steps, indices = chk.get_timesteps()
            t.assign(steps[-1])
            iplot = indices[-1]
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
    if True:
        coords = mesh_f.coordinates.dat.data_with_halos
        coords[:] = coords[:] + solution.subfunctions[2].dat.data_ro_with_halos[:]
        pgfplot(solution.subfunctions[4], "pressure.dat", degree=2)
        #Vplot = VectorFunctionSpace(mesh_f, "CG", 1)
        #fplot = Function(Vplot).interpolate(as_vector([x_f, y_f]))
        #pgfplot(fplot, "quiver.dat", degree=0)
        raise RuntimeError("only plotted solution")
    ii = 0
    while float(t) < T:
        t.assign(float(t) + float(dt))
        ii += 1
        if mesh.comm.rank == 0:
            print(f"Computing solution at time = {float(t)} (dt = {float(dt)}, CNshift={CNshift})", flush=True)
        solver.solve()
        for subfunction, subfunction_0 in zip(solution.subfunctions, solution_0.subfunctions):
            subfunction_0.assign(subfunction)
        # Everything is now up to date.
        FD = assemble(dot(sigma_f_, dot(J_f_ * transpose(inv(F_f_)), n_f))[0] * ds_f((label_circle, label_interface)))
        FL = assemble(dot(sigma_f_, dot(J_f_ * transpose(inv(F_f_)), n_f))[1] * ds_f((label_circle, label_interface)))
        u_A = solution.subfunctions[3].at(pointA, tolerance=1.e-6)
        if mesh.comm.rank == 0:
            print(f"FD     = {FD}")
            print(f"FL     = {FL}")
            print(f"uA     = {u_A}")
            if ii % 10 == 0:
                with open(fname_FD, 'a') as outfile:
                    outfile.write(f"{float(t)} {FD}" + "\n")
                with open(fname_FL, 'a') as outfile:
                    outfile.write(f"{float(t)} {FL}" + "\n")
                    #sample_FD[itimestep // (ntimesteps // nsample)] = FD
                    #sample_FL[itimestep // (ntimesteps // nsample)] = FL
                    #np.savetxt(outfile, np.concatenate([sample_t.reshape(-1, 1), sample_FD.reshape(-1, 1)], axis=1))
        if ii % 10 == 0:
            iplot += 1
            with DumbCheckpoint(fname_checkpoint, mode=FILE_UPDATE) as chk:
                chk.set_timestep(float(t), iplot)
                for subsolution in solution.subfunctions:
                    chk.store(subsolution)

    end = time.time()
    print(f"Time: {end - start}")
else:
    raise ValueError

