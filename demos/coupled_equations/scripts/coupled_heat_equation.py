from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import trisurf
import numpy as np

# Constants
PLOT = False
VERBOSE = True

dt = 0.001
n_steps = 100
T = n_steps * dt

# Diffusion Coefficients
kappa1 = Constant(1.0)
kappa2 = Constant(1.0)

# Variables initialised for convergence analysis
n1_list = [2,4,8,16]
n2_list = [2,4,8,16]
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

def build_problem(mesh1, mesh2):
    p = 4
    p_inner = 2
    w2 = Constant(50.0)/CellDiameter(mesh2)  # Nitsche penalty weight

    # Spatial Coordinates
    x1, y1 = SpatialCoordinate(mesh1)
    x2, y2 = SpatialCoordinate(mesh2)
    t = Constant(0.0)

    # Exact Solutions
    phi1 = x1 * sin(pi * y1)**2
    phi2 = (sin(pi * y2)**2 * (x2 - (x2 - 1.0)**2 / 2.0))
    u1_exact = exp(-t) * phi1
    u2_exact = exp(-t) * phi2
    
    # RHS functions
    f1 = -exp(-t) * phi1 - div(grad(u1_exact))
    f2 = -exp(-t) * phi2 - div(grad(u2_exact))

    n1 = FacetNormal(mesh1)
    n2 = FacetNormal(mesh2)
    dx1 = Measure("dx", domain=mesh1)
    dx2 = Measure("dx", domain=mesh2)
    ds1 = Measure("ds", domain=mesh1, subdomain_id=2)
    ds2 = Measure("ds", domain=mesh2, subdomain_id=1)

    # Function Spaces
    V1 = FunctionSpace(mesh1, "CG", p)
    V2 = FunctionSpace(mesh2, "CG", p)

    # Intermediate spaces
    Q1v = VectorFunctionSpace(mesh1, "CG", p_inner)
    Q2 = FunctionSpace(mesh2, "CG", p_inner)

    W = V1 * V2
    # Test and trial functions
    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    # Previous time solution
    u_old = Function(W)
    u1_old, u2_old = split(u_old)

    # Poisson on mesh_1
    A11_form = inner(u1/dt, v1) * dx1 + kappa1 * inner(grad(u1), grad(v1)) * dx1

    # Helmholtz on mesh_2
    A22_form = inner(u2/dt, v2) * dx2 + kappa2 * inner(grad(u2), grad(v2)) * dx2 \
                - inner(dot(grad(u2), n2), v2) * ds2 \
                - inner(dot(grad(v2), n2), u2) * ds2 \
                + w2 * inner(u2, v2) * ds2
    
    # A12: row v1, column u2
    q1v = TrialFunction(Q1v)
    M1 = -inner(dot(q1v, n1), v1) * ds1
    B12 = interpolate(grad(u2), Q1v, allow_missing_dofs=True)
    A12_form = action(M1, B12) 

    # A21: row v2, column u1.
    q2 = TrialFunction(Q2)
    M2 = -w2 * inner(q2, v2) * ds2 + inner(q2, dot(grad(v2), n2)) * ds2
    B21 = interpolate(u1, Q2, allow_missing_dofs=True)
    A21_form = action(M2, B21)

    # RHS
    b1 = inner(f1, v1) * dx1 + inner(u1_old/dt, v1) * dx1
    b2 = inner(f2, v2) * dx2 + inner(u2_old/dt, v2) * dx2
    A = A11_form + A12_form + A21_form + A22_form
    L = b1 + b2

    # Exact solutions used for further analysis
    u1_exact_func = Function(V1).interpolate(u1_exact)
    u2_exact_func = Function(V2).interpolate(u2_exact)

    return A, L, W, u_old, t, u1_exact_func, u2_exact_func

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
    plt.close(fig)

# Solver for the coupled problem for each defined mesh
for n1, n2, mesh1, mesh2 in zip(n1_list, n2_list, mesh1_list, mesh2_list):
    print(f"Solving n1={n1}, n2={n2}")

    A, L, W, u_old, t, u1_exact_func, u2_exact_func = build_problem(mesh1, mesh2)
    u_sol = Function(W)

    u1_old, u2_old = u_old.subfunctions
    u1_old.interpolate(u1_exact_func)
    u2_old.interpolate(u2_exact_func)


    bc1 = DirichletBC(W.sub(0), 0, [1, 3, 4])
    #bc2 = DirichletBC(W.sub(1), 0, [2, 3, 4])
    problem = LinearVariationalProblem(A, L, u_sol, bcs=bc1)
    params = {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
    solver = LinearVariationalSolver(problem, solver_parameters=params)

    for step in range(n_steps):
        current_time = (step+1) * dt
        t.assign(current_time)
        solver.solve()
        u_old.assign(u_sol)

    u_1, u_2 = u_sol.subfunctions

    if PLOT:
        plot(f"heat_coupled_{n1}_{n2}.png", u_1, u_2)

    # Calculates the L2 error between the approximated and exact solutions
    e_1 = errornorm(u1_exact_func, u_1, norm_type="L2")
    e_2 = errornorm(u2_exact_func, u_2, norm_type="L2")
    h1 = 1.0/n1
    h2 = 1.0/n2

    h1_array.append(h1)
    h2_array.append(h2)
    errors_1.append(e_1)
    errors_2.append(e_2)

# Calculate the convergence rate of the approcimated solution
ratios_1 = []
ratios_2 = []
for i in range(len(h1_array) - 1):
    q1_numerator = np.log(errors_1[i]/errors_1[i+1])
    q2_numerator = np.log(errors_2[i]/errors_2[i+1])
    q1_denominator = np.log(h1_array[i]/h1_array[i+1])
    q2_denominator = np.log(h2_array[i]/h2_array[i+1])

    q1 = q1_numerator/q1_denominator
    q2 = q2_numerator/q2_denominator
    ratios_1.append(q1)
    ratios_2.append(q2)

# Outputs convergence analysis results and presents a log-log graph
if VERBOSE:
    print(f"{'h':>10} {'Error 1':>15} {'Rate 1':>10}")
    for i in range(len(errors_1)):
        if i == 0:
            print(f"{h1_array[i]:10.5f} {errors_1[i]:15.6e} {'-':>10}")
        else:
            print(f"{h1_array[i]:10.5f} {errors_1[i]:15.6e} {ratios_1[i-1]:10.4f}")

    print(f"{'h':>10} {'Error 2':>15} {'Rate 2':>10}")
    for i in range(len(errors_2)):
        if i == 0:
            print(f"{h2_array[i]:10.5f} {errors_2[i]:15.6e} {'-':>10}")
        else:
            print(f"{h2_array[i]:10.5f} {errors_2[i]:15.6e} {ratios_2[i-1]:10.4f}")

# PETSc.Sys.Print(f"...")

    plt.figure(figsize=(8,8))
    plt.loglog(h2_array, errors_2, "o-", label="Mesh 2")
    plt.loglog(h1_array, errors_1, "s-", label="Mesh 1")
    plt.xlabel("h")
    plt.ylabel("L2 error")
    plt.gca().invert_xaxis()
    plt.grid(False)
    plt.legend()
    plt.title("Heat Equations Coupled with Nitsche's Method")
    plt.savefig("Logloggraph.png")
