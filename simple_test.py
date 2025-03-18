from firedrake import *

parameters = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.OffloadPC",
    "offload": {
        "pc_type": "ksp",
        "ksp": {
            "ksp_type": "cg",
            "ksp_view": None,
            "ksp_rtol": "1e-10",
            "ksp_monitor": None,
            "pc_type": "sor",
        }
    }
}

mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate(2*pi**2*sin(pi*x)*sin(pi*y))

# Equations
L = inner(grad(u), grad(v)) * dx
R = inner(v, f) * dx

# Dirichlet boundary on all sides to 0
bcs = DirichletBC(V, 0, "on_boundary")

# Exact solution
sol = Function(V)
sol.interpolate(sin(pi*x)*sin(pi*y))

# Solution function
u_f = Function(V)

problem = LinearVariationalProblem(L, R, u_f, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=parameters)
solver.solve()
errornorm(problem.u, sol)
