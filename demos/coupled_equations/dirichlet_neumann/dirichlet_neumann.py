from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import trisurf
import numpy as np


# Tight two-way coupling w/ Dirichlet-Neumann 
# Poisson on mesh_1
# Helmholtz on mesh_2
# mesh_1 receives the flux of u2 
# mesh_2 receives the trace of u1

PLOT = False
h_array = []
errors_1 = []
errors_2 = []

n_list = [2, 4, 8, 16, 32, 64]

def setup_and_solve(mesh_1, mesh_2):
    x1, y1 = SpatialCoordinate(mesh_1)
    x2, y2 = SpatialCoordinate(mesh_2)

    V1 = FunctionSpace(mesh_1, "CG", 3)
    V2 = FunctionSpace(mesh_2, "CG", 3)

    # intermediate spaces
    Q1v = VectorFunctionSpace(mesh_1, "CG", 3)
    Q2 = FunctionSpace(mesh_2, "CG", 3)

    W = V1 * V2

    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    u1_exact = x1 * sin(pi * y1) ** 2
    u2_exact = sin(pi * y2) ** 2 * (x2 - (x2 - 1) ** 2 / 2)

    # RHS functions
    f1 = -div(grad(u1_exact))
    f2 = -div(grad(u2_exact)) + u2_exact

    # Nitsche penalty weights
    w2 = Constant(100.0)

    n1 = FacetNormal(mesh_1)
    n2 = FacetNormal(mesh_2)

    dx1 = Measure("dx", domain=mesh_1)
    dx2 = Measure("dx", domain=mesh_2)
    ds1 = Measure("ds", domain=mesh_1, subdomain_id=2)
    ds2 = Measure("ds", domain=mesh_2, subdomain_id=1)

    # Poisson on mesh_1
    A11_form = inner(grad(u1), grad(v1)) * dx1

    # Helmholtz on mesh_2
    A22_form = (inner(grad(u2), grad(v2)) + inner(u2, v2)) * dx2 \
            - inner(dot(grad(u2), n2), v2) * ds2 \
            + w2 * inner(u2, v2) * ds2

    # A12: row v1, column u2
    # W --B12--> Q1v --M1--> W^*
    # inner(dot(grad(u2), n1), v1) * ds1
    q1v = TrialFunction(Q1v)
    M1 = -inner(dot(q1v, n1), v1) * ds1  # Q1v -> W^*
    B12 = interpolate(grad(u2), Q1v, allow_missing_dofs=True)  # W -> Q1v
    A12_form = action(M1, B12)

    # A21: row v2, column u1.
    # W --B21--> Q2 --M2--> W^*
    q2 = TrialFunction(Q2)
    M2 = -w2 * inner(q2, v2) * ds2  # Q2 -> W^*
    B21 = interpolate(u1, Q2, allow_missing_dofs=True)  # W -> Q2
    A21_form = action(M2, B21)

    # RHS
    b1 = inner(f1, v1) * dx1
    b2 = inner(f2, v2) * dx2

    bc = DirichletBC(W.sub(0), 0, [1, 3, 4])

    A = A11_form + A12_form + A21_form + A22_form
    L = b1 + b2

    u_sol = Function(W)
    problem = LinearVariationalProblem(A, L, u_sol, bcs=bc)
    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    u_1, u_2 = u_sol.subfunctions
    return u_1, u_2, V1, V2, u1_exact, u2_exact, dx1, dx2

def plot(filename, u_1, u_2):
    u1_vals = u_1.dat.data_ro
    u2_vals = u_2.dat.data_ro
    vmin = min(u1_vals.min(), u2_vals.min())
    vmax = max(u1_vals.max(), u2_vals.max())

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")
    trisurf(u_1, axes=ax, vmin=vmin, vmax=vmax, cmap="viridis")
    trisurf(u_2, axes=ax, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.view_init(elev=35, azim=-110)
    ax.set_aspect("equalxz")
    plt.tight_layout()
    plt.savefig(filename)
    

for i in range(len(n_list)):
    n = n_list[i]
    mesh_2 = UnitSquareMesh(n, n, quadrilateral=True)
    mesh_2.coordinates.dat.data[:, 0] += 1.0  # Shift to the right by 1
    mesh_1 = UnitSquareMesh(n, n, quadrilateral=True)

    u_1, u_2, V1, V2, u1_exact, u2_exact, dx1, dx2 = setup_and_solve(mesh_1, mesh_2)

    if PLOT:
        plot(f"demos/coupled_equations/dirichlet_neumann/dirichlet_neumann_example_{i}.png", u_1, u_2)

    u1_exact_func = Function(V1).interpolate(u1_exact)
    u2_exact_func = Function(V2).interpolate(u2_exact)

    e_1 = errornorm(u1_exact_func, u_1, norm_type="L2")
    e_2 = errornorm(u2_exact_func, u_2, norm_type="L2")
    h = 1/n

    h_array.append(h)
    errors_1.append(e_1)
    errors_2.append(e_2)

# Error term - O(h^(n+1))

ratios_1 = []
ratios_2 = []
for i in range(len(h_array) - 1):
    q1_numerator = np.log(errors_1[i]/errors_1[i+1])
    q2_numerator = np.log(errors_2[i]/errors_2[i+1])
    q_denominator = np.log(h_array[i]/h_array[i+1])

    q1 = q1_numerator/q_denominator
    q2 = q2_numerator/q_denominator
    ratios_1.append(q1)
    ratios_2.append(q2)


print(f"{'h':>10} {'Error 1':>15} {'Rate 1':>10}")

for i in range(len(errors_1)):
    if i == 0:
        print(f"{h_array[i]:10.5f} {errors_1[i]:15.6e} {'-':>10}")
    else:
        print(f"{h_array[i]:10.5f} {errors_1[i]:15.6e} {ratios_1[i-1]:10.4f}")

plt.figure(figsize=(8,8))
plt.loglog(h_array, errors_1, "o-", label="Poisson")
plt.loglog(h_array, errors_2, "s-", label="Helmholtz")
plt.xlabel("h")
plt.ylabel("L2 error")
plt.gca().invert_xaxis()
plt.grid(False)
plt.legend()
plt.savefig("demos/coupled_equations/dirichlet_neumann/Logloggraph.png")