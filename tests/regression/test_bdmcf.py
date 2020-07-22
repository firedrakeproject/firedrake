from firedrake import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(16, 16, quadrilateral=True)
x = SpatialCoordinate(mesh)

BDM = FunctionSpace(mesh, "BDMCF", 2)
DPC = FunctionSpace(mesh, "DPC", 1)
W = BDM * DPC

# Define trial and test functions
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

# Define source function
f = Function(DPC).interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

# Define variational form
a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx(degree=10)
L = -f*v*dx(degree=10)

# Compute solution
w = Function(W)
sp = {'pc_type': 'fieldsplit',
      'pc_fieldsplit_type': 'schur',
      'ksp_type': 'preonly',
      'pc_fieldsplit_schur_fact_type': 'FULL',
      'fieldsplit_0_ksp_type': 'cg',
      'fieldsplit_0_pc_factor_shift_type': 'INBLOCKS',
      'fieldsplit_1_pc_factor_shift_type': 'INBLOCKS',
      'fieldsplit_1_ksp_type': 'cg'}

solve(a == L, w, solver_parameters=sp)
# solve(a == L, w)
sigma, u = w.split()

# Analytical solution
f.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))

l2err = errornorm(f, u, norm_type="l2")
print(l2err)

plt.figure(1)
plot(u)

plt.figure(2)
plot(f)

plt.show()
