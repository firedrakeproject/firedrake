from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import trisurf
import numpy as np


# Tight two-way coupling

PLOT = True
#REFINEMENT_LEVELS = 3
h_array = []
errors_1 = []
errors_2 = []

n_list = [8, 16, 32, 38, 45, 50, 75, 100]

# Mesh Refinement
#meshes_1 = MeshHierarchy(mesh_1, refinement_levels=REFINEMENT_LEVELS)
#meshes_2 = MeshHierarchy(mesh_2, refinement_levels=REFINEMENT_LEVELS)

def setup_and_solve(mesh_1, mesh_2):
    x1, y1 = SpatialCoordinate(mesh_1)
    x2, y2 = SpatialCoordinate(mesh_2)

    V1 = FunctionSpace(mesh_1, "CG", 1)
    V2 = FunctionSpace(mesh_2, "CG", 1)

    # intermediate spaces
    Q1 = FunctionSpace(mesh_1, "CG", 2)
    Q2 = FunctionSpace(mesh_2, "CG", 2)

    W = V1 * V2

    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    # RHS functions
    f1 = Function(V1).interpolate(((16 * pi**2 + 1) * (y1 - 1)**2 * y1*y1 - 12*y1*y1 + 12*y1 - 2) * cos(4 * pi * x1))
    f2 = Function(V2).interpolate((-2 + 12*y2 - 12*y2*y2 + pi**2 * y2**2 * ((1-y2)**2/4)) * cos((pi/2) * (x2 - 1)))

    # Nitsche penalty weights
    w1 = Constant(100.0)
    w2 = Constant(100.0)

    n1 = FacetNormal(mesh_1)
    n2 = FacetNormal(mesh_2)

    dx1 = Measure("dx", domain=mesh_1)
    dx2 = Measure("dx", domain=mesh_2)
    ds1 = Measure("ds", domain=mesh_1, subdomain_id=2)
    ds2 = Measure("ds", domain=mesh_2, subdomain_id=1)

    # Helmholtz on mesh_1.
    A11_form = (inner(grad(u1), grad(v1)) + inner(u1, v1)) * dx1 \
            - inner(dot(grad(u1), n1), v1) * ds1 \
            + w1 * inner(u1, v1) * ds1

    # Poisson on mesh_2.
    A22_form = inner(grad(u2), grad(v2)) * dx2 \
            - inner(dot(grad(u2), n2), v2) * ds2 \
            + w2 * inner(u2, v2) * ds2

    # A12: row v1, column u2.
    # W --B12--> Q1 --M1--> W^*
    q1 = TrialFunction(Q1)
    M1 = -w1 * inner(q1, v1) * ds1  # Q1 -> W^*
    B12 = interpolate(u2, Q1, allow_missing_dofs=True)  # W -> Q1
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

    bc = DirichletBC(W.sub(1), 0, [1, 3, 4])

    A = A11_form + A12_form + A21_form + A22_form
    L = b1 + b2

    u_sol = Function(W)
    problem = LinearVariationalProblem(A, L, u_sol, bcs=bc)
    solver = LinearVariationalSolver(problem)
    solver.solve()

    u_1, u_2 = u_sol.subfunctions
    return u_1, u_2, V1, V2, x1, y1, x2, y2

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

    u_1, u_2, V1, V2, x1, y1, x2, y2 = setup_and_solve(mesh_1, mesh_2)

    if PLOT:
        plot(f"example_{i}.png", u_1, u_2)

    u1_exact = Function(V1).interpolate(cos(4 * pi * x1) * y1 * y1 * (1 - y1)**2)
    u2_exact = Function(V2).interpolate(cos((x2 - 1) * (pi/2)) * y2 * y2 * (1 - y2)**2)

    e_1 = errornorm(u_1, u1_exact, norm_type="L2")
    e_2 = errornorm(u_2, u2_exact, norm_type="L2")
    #e_1 = sqrt(assemble(dot(u_1 - u1_exact, u_1 - u1_exact) * dx1)) 
    #e_2 = sqrt(assemble(dot(u_2 - u2_exact, u_2 - u2_exact) * dx2))
    h = 1/n

    h_array.append(h)
    errors_1.append(e_1)
    errors_2.append(e_2)

print("h values", h_array)
print("error values 1", errors_1)
print("error values 2", errors_2)
# Error term - O(h^(n+1))

ratios_1 = []
ratios_2 = []
for i in range(len(h_array) - 1):
    q1_numerator = ln(errors_1[i]/errors_1[i+1])
    q2_numerator = ln(errors_2[i]/errors_2[i+1])
    q_denominator = ln(h_array[i]/h_array[i+1])

    q1 = q1_numerator/q_denominator
    q2 = q2_numerator/q_denominator
    ratios_1.append(q1)
    ratios_2.append(q2)

print("ratios 1", ratios_1)
print("ratios 2", ratios_2)

plt.figure(figsize=(8,8))
plt.loglog(h_array, errors_1, "o-", label="Helmholtz")
plt.loglog(h_array, errors_2, "s-", label="Poisson")
plt.xlabel("h")
plt.ylabel("L2 error")
plt.gca().invert_xaxis()
plt.grid(False)
plt.legend()
plt.savefig("Logloggraph.png")