from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import trisurf
import numpy as np


# Tight two-way coupling w/ Dirichlet-Neumann 
# Poisson on mesh_1
# Helmholtz on mesh_2
# mesh_1 receives the flux of u2 
# mesh_2 receives the trace of u1

# Constants
PLOT = True
VERBOSE = True

# Variables initialised for convergence analysis
n1_list = [8,8,8,8,8] #[4,4,4,4,4]#[8,8,8,8,8] #
n2_list = [2,4,8,16,32]
mesh1_list = []
mesh2_list = []
h1_array = []
h2_array = []
errors_1 = []
errors_2 = []

# Prepares meshes with differing refinement levels for convergence analysis
for n1,n2 in zip(n1_list, n2_list):
    mesh1 = RectangleMesh(nx = n1, ny = n1//2, Lx = 0.5, Ly = 1)
    mesh2 = RectangleMesh(nx = n2, ny = n2//2, Lx = 0.5, Ly = 1)
    mesh2.coordinates.dat.data[:, 0] += 0.5  # Shift to the right by 0.5
    
    coords = mesh2.coordinates.dat.data
    coords[np.isclose(coords[:, 0], 0.5), 0] = 0.5
    
    mesh1_list.append(mesh1)
    mesh2_list.append(mesh2)

def build_problem(mesh1, mesh2):
    p = 5
    p_inner = 2
    w1 = Constant(1000.0)/CellDiameter(mesh1)
    w2 = Constant(1000.0)/CellDiameter(mesh2)  # Nitsche penalty weight

    x1, y1 = SpatialCoordinate(mesh1)
    x2, y2 = SpatialCoordinate(mesh2)
    u1_exact = x1*(0.5-x1)*(1-x1)*sin(pi*y1)
    u2_exact = x2*(0.5-x2)*(1-x2)*sin(pi*y2)
    
    # RHS functions
    f1 = -div(grad(u1_exact))
    f2 = -div(grad(u2_exact))

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
    Q1 = FunctionSpace(mesh1, "CG", p_inner)
    Q2 = FunctionSpace(mesh2, "CG", p_inner)

    W = V1 * V2
    # Test and trial functions
    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    # Poisson on mesh_1
    A11_form = inner(grad(u1), grad(v1)) * dx1 \
                + w1 * inner(v1, u1) * ds1 \
                - inner(dot(grad(u1), n1), v1) * ds1 \
                - inner(dot(grad(v1), n1), u1) * ds1

    # Poisson on mesh_2  (drop the +u2_exact in f2 too, unless Helmholtz is intended)
    A22_form = inner(grad(u2), grad(v2)) * dx2 \
                + w2 * inner(v2, u2) * ds2 \
                - inner(dot(grad(u2), n2), v2) * ds2 \
                - inner(dot(grad(v2), n2), u2) * ds2

    q1 = TrialFunction(Q1)
    M1_w = -w1 * inner(q1, v1) * ds1
    M1_trace = inner(dot(grad(v1), n1), q1) * ds1
    B12 = interpolate(u2, Q1, allow_missing_dofs=True)
    A12_w = action(M1_w, B12)
    A12_trace = action(M1_trace, B12)

    q2 = TrialFunction(Q2)
    M2_w = -w2 * inner(q2, v2) * ds2
    M2_trace = inner(dot(grad(v2), n2), q2) * ds2
    B21 = interpolate(u1, Q2, allow_missing_dofs=True)
    A21_w = action(M2_w, B21)
    A21_trace = action(M2_trace, B21)

    A = A11_form + A22_form + A12_w + A12_trace + A21_w + A21_trace

    b1 = inner(f1, v1) * dx1
    b2 = inner(f2, v2) * dx2
    L = b1 + b2

    # Exact solutions used for further analysis
    u1_exact_func = Function(V1).interpolate(u1_exact)
    u2_exact_func = Function(V2).interpolate(u2_exact)

    return A, L, W, u1_exact_func, u2_exact_func

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

# Solver for the coupled problem for each defined mesh
for n1, n2, mesh1, mesh2 in zip(n1_list, n2_list, mesh1_list, mesh2_list):
    A, L, W, u1_exact_func, u2_exact_func = build_problem(mesh1, mesh2)
    u_sol = Function(W)
    
    bc1 = DirichletBC(W.sub(0), 0, [1, 3, 4])
    bc2 = DirichletBC(W.sub(1), 0, [2, 3, 4])
    problem = LinearVariationalProblem(A, L, u_sol, bcs=[bc1, bc2])
    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    u_1, u_2 = u_sol.subfunctions

    if PLOT:
        plot(f"mortar_dirichlet_example_{n1}_{n2}.png", u_1, u_2)

    # Calculates the L2 error between the approximated and exact solutions
    e_1 = errornorm(u1_exact_func, u_1, norm_type="L2")
    e_2 = errornorm(u2_exact_func, u_2, norm_type="L2")
    h1 = 1.0/n1
    h2 = 1.0/n2
    #h1 = CellDiameter(mesh1)
    #h2 = CellDiameter(mesh2)

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
    plt.loglog(h2_array, errors_2, "o-", label="Poisson")
    #plt.loglog(h1_array, errors_1, "s-", label="Poisson")
    plt.xlabel("h")
    plt.ylabel("L2 error")
    plt.gca().invert_xaxis()
    plt.grid(False)
    plt.legend()
    plt.title("Poisson-Poisson Coupling with Dirichlet BCs using the Mortar Method")
    plt.savefig("Logloggraph.png")