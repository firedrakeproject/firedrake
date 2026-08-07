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
PLOT = False
VERBOSE = True

# Variables initialised for convergence analysis
n_list = [2,5,17,21,41]
mesh_list = []
h_array = []
errors = []

# Prepares meshes with differing refinement levels for convergence analysis
for n in n_list:
    mesh = UnitSquareMesh(n, n, quadrilateral=True)
    mesh_list.append(mesh)

def build_problem(mesh):
    p = 3
    w = Constant(100.0)/CellDiameter(mesh)

    x, y = SpatialCoordinate(mesh)
    u_exact = x*(1-x)*sin(pi*y)

    # RHS functions
    f = -div(grad(u_exact))
    n = FacetNormal(mesh)
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)

    # Function Space
    V = FunctionSpace(mesh, "CG", p)

    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)

    A = dot(grad(v), grad(u)) * dx - inner(v, dot(n, grad(u))) * ds + w * inner(v,u) * ds
    L = f * v * dx

    # Exact solutions used for further analysis
    #u_exact_func = Function(V).interpolate(u_exact)

    return A, L, V, u_exact

def plot(filename, u):
    u_vals = u.dat.data_ro
    vmin = u_vals.min()
    vmax = u_vals.max()

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")
    trisurf(u, axes=ax, vmin=vmin, vmax=vmax, cmap="viridis")
    #ax.view_init(elev=35, azim=-110)
    #ax.set_aspect("equalxz")
    plt.tight_layout()
    plt.savefig(filename)

# Solver for the coupled problem for each defined mesh
for n, mesh in zip(n_list, mesh_list):
    A, L, V, u_exact_func = build_problem(mesh)
    u_sol = Function(V)
    
    solve(A == L, u_sol)
    u = u_sol

    if PLOT:
        plot(f"poisson_example_{n}.png", u)

    # Calculates the L2 error between the approximated and exact solutions
    e_1 = errornorm(u_exact_func, u, norm_type="L2")
    h1 = 1.0/n

    h_array.append(h1)
    errors.append(e_1)

# Calculate the convergence rate of the approcimated solution
ratios_1 = []
for i in range(len(h_array) - 1):
    q1_numerator = np.log(errors[i]/errors[i+1])
    q1_denominator = np.log(h_array[i]/h_array[i+1])

    q1 = q1_numerator/q1_denominator
    ratios_1.append(q1)

# Outputs convergence analysis results and presents a log-log graph
if VERBOSE:
    print(f"{'h':>10} {'Error 1':>15} {'Rate 1':>10}")
    for i in range(len(errors)):
        if i == 0:
            print(f"{h_array[i]:10.5f} {errors[i]:15.6e} {'-':>10}")
        else:
            print(f"{h_array[i]:10.5f} {errors[i]:15.6e} {ratios_1[i-1]:10.4f}")

# PETSc.Sys.Print(f"...")

    plt.figure(figsize=(8,8))
    plt.loglog(h_array, errors, "s-", label="Poisson")
    plt.xlabel("h")
    plt.ylabel("L2 error")
    plt.gca().invert_xaxis()
    plt.grid(False)
    plt.legend()
    plt.title("Helmholtz-Poisson Coupling with Dirichlet-Neumann Method")
    plt.savefig("Logloggraph.png")