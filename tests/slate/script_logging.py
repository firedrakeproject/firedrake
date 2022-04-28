from firedrake import *


mesh = UnitSquareMesh(2, 2)
U = FunctionSpace(mesh, "RT", 1)
V = FunctionSpace(mesh, "DG", 0)
W = U * V
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
n = FacetNormal(mesh)

# Define the source function
x, y = SpatialCoordinate(mesh)
f = Function(V)
f.interpolate(10*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

# Define the variational forms
a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx
L = -inner(f, v) * dx + Constant(0.0) * dot(conj(tau), n) * (ds(3) + ds(4))

# Compare hybridized solution with non-hybridized
w = Function(W)
bc1 = DirichletBC(W[0], as_vector([0.0, -sin(5*x)]), 1)
bc2 = DirichletBC(W[0], as_vector([0.0, sin(5*y)]), 2)
bcs = [bc1, bc2]

params = {'mat_type': 'matfree',
          'ksp_type': 'preonly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.HybridizationPC',
          'hybridization': {'ksp_type': 'cg',
                            'pc_type': 'jacobi',
                            'ksp_rtol': 1e-8}}

solve(a == L, w, bcs=bcs, solver_parameters=params)
