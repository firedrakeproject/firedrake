from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import trisurf
import numpy as np
from irksome import GaussLegendre, Dt, TimeStepper

# Constants
PLOT = False
VERBOSE = True

# Variables initialised for convergence analysis
n1_list = [16,16,16,16,16]
n2_list = [2,4,8,16,32]
mesh1_list = []
mesh2_list = []
h1_array = []
h2_array = []
errors_1 = []
errors_2 = []

# Prepares meshes with differing refinement levels for convergence analysis
for n1,n2 in zip(n1_list, n2_list):
    mesh1 = UnitSquareMesh(n1, n1, quadrilateral=True)
    mesh2 = UnitSquareMesh(n2, n2, quadrilateral=True)
    mesh2.coordinates.dat.data[:, 0] += 1.0  # Shift to the right by 1

    mesh1_list.append(mesh1)
    mesh2_list.append(mesh2)

def build_problem(mesh1, mesh2, W, u_sol):
    p_inner = 2
    w2 = Constant(50.0)/CellDiameter(mesh2)

    n1 = FacetNormal(mesh1)
    n2 = FacetNormal(mesh2)
    dx1 = Measure("dx", domain=mesh1)
    dx2 = Measure("dx", domain=mesh2)
    ds1 = Measure("ds", domain=mesh1, subdomain_id=2)
    ds2 = Measure("ds", domain=mesh2, subdomain_id=1)

    Q1v = VectorFunctionSpace(mesh1, "CG", p_inner)
    Q2 = FunctionSpace(mesh2, "CG", p_inner)

    u1, u2 = split(u_sol)
    v1, v2 = TestFunctions(W)

    A11_form = inner(Dt(u1), v1) * dx1 + inner(grad(u1), grad(v1)) * dx1

    A22_form = inner(Dt(u2), v2) * dx2 + inner(grad(u2), grad(v2)) * dx2 \
                - inner(dot(grad(u2), n2), v2) * ds2 \
                - inner(dot(grad(v2), n2), u2) * ds2 \
                + w2 * inner(u2, v2) * ds2

    u2_flux_on_1 = interpolate(grad(u2), Q1v, allow_missing_dofs=True)
    A12_form = -inner(dot(u2_flux_on_1, n1), v1) * ds1

    u1_trace_on_2 = interpolate(u1, Q2, allow_missing_dofs=True)
    A21_form = -w2 * inner(u1_trace_on_2, v2) * ds2 + inner(u1_trace_on_2, dot(grad(v2), n2)) * ds2

    F = A11_form + A12_form + A21_form + A22_form
    return F

def plot(filename, u_1, u_2):
    u1_vals = u_1.dat.data_ro
    u2_vals = u_2.dat.data_ro
    vmin = min(u1_vals.min(), u2_vals.min())
    vmax = max(u1_vals.max(), u2_vals.max())

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")
    trisurf(u_1, axes=ax, vmin=vmin, vmax=vmax, cmap="viridis")
    trisurf(u_2, axes=ax, vmin=vmin, vmax=vmax, cmap="viridis")
    #ax.view_init(elev=35, azim=-110)
    #ax.set_aspect("equalxz")
    plt.tight_layout()
    plt.savefig(filename)

# Irksome: create Butcher tableau
butcher_tableau = GaussLegendre(2)  # aka implicit midpoint rule
ns = butcher_tableau.num_stages

# Define variables to store time step and current time
n_steps = 200
t_stepsize = 0.001
initial_time = 0.0

params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu"
        }

# Solver for the coupled problem for each defined mesh
for n1, n2, mesh1, mesh2 in zip(n1_list, n2_list, mesh1_list, mesh2_list):
    V1 = FunctionSpace(mesh1, "CG", 4)
    V2 = FunctionSpace(mesh2, "CG", 4)
    W = V1 * V2
    u_sol = Function(W)

    x1, y1 = SpatialCoordinate(mesh1)
    x2, y2 = SpatialCoordinate(mesh2)
    x0, y0, sigma = 0.5, 0.5, 0.1
    u_sol.sub(0).interpolate(0.5*exp(-((x1-x0)**2+(y1-y0)**2)/(2*sigma**2)))
    u_sol.sub(1).interpolate(0.5*exp(-((x2-x0)**2+(y2-y0)**2)/(2*sigma**2)))

    #F = build_problem(mesh1, mesh2, W, u_sol)
    p_inner = 2
    w2 = Constant(50.0)/CellDiameter(mesh2)

    n1 = FacetNormal(mesh1)
    n2 = FacetNormal(mesh2)
    ds1 = Measure("ds", domain=W.mesh(),
              intersect_measures=[Measure(ds(mesh1))])
    ds2 = Measure("ds", domain=W.mesh(),
              intersect_measures=[Measure(ds(mesh2))])
    dx1 = Measure("dx", domain=W.mesh(),
              intersect_measures=[Measure("dx", domain=mesh1)])
    dx2 = Measure("dx", domain=W.mesh(),
              intersect_measures=[Measure("dx", domain=mesh2)])

    Q1v = VectorFunctionSpace(mesh1, "CG", p_inner)
    Q2 = FunctionSpace(mesh2, "CG", p_inner)

    u1, u2 = split(u_sol)
    v1, v2 = TestFunctions(W)

    A11_form = inner(Dt(u1), v1) * dx1 + inner(grad(u1), grad(v1)) * dx1

    A22_form = inner(Dt(u2), v2) * dx2 + inner(grad(u2), grad(v2)) * dx2 \
                - inner(dot(grad(u2), n2), v2) * ds2 \
                - inner(dot(grad(v2), n2), u2) * ds2 \
                + w2 * inner(u2, v2) * ds2

    u2_flux_on_1 = Function(Q1v)   # holds grad(u2)·, transferred, at the *previous* time level
    u1_trace_on_2 = Function(Q2)   # holds u1, transferred, at the *previous* time level

    A12_form = -inner(dot(u2_flux_on_1, n1), v1) * ds1
    A21_form = -w2 * inner(u1_trace_on_2, v2) * ds2 + inner(u1_trace_on_2, dot(grad(v2), n2)) * ds2

    F = A11_form + A12_form + A21_form + A22_form

    bc1 = DirichletBC(W.sub(0), 0, [1, 3, 4])
    bc2 = DirichletBC(W.sub(1), 0, [2, 3, 4])  

    R1_space = FunctionSpace(mesh1, "R", 0)
    dt1 = Function(R1_space).assign(t_stepsize)
    t1 = Function(R1_space).assign(initial_time)

    stepper = TimeStepper(F, butcher_tableau, t1, dt1, u_sol,
                           bcs=[bc1, bc2], solver_parameters=params)


    #times: list[float] = [0.0]

    #solutions: list[np.ndarray] = [u_sol.dat.data_ro.copy()]
    for step in range(n_steps):
        u1_now, u2_now = u_sol.subfunctions   # genuine per-mesh Functions — unambiguous domain
        u2_flux_on_1.interpolate(grad(u2_now), allow_missing_dofs=True)
        u1_trace_on_2.interpolate(u1_now, allow_missing_dofs=True)

        stepper.advance()
        print(f"t = {float(t1):.4f}")
        t1.assign(float(t1) + float(dt1))
        #times.append(float(t1))

        print(u_sol.dat.data_ro)
        #solutions.append(u_sol.dat.data_ro.copy())

    #print(solutions)
