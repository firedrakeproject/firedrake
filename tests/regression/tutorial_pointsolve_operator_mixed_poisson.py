from firedrake import *
import matplotlib.pyplot as plt

nn = 32
mesh = UnitSquareMesh(nn, nn)

BDM = FunctionSpace(mesh, "BDM", 1)
DG = FunctionSpace(mesh, "DG", 0)
W = BDM * DG

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
i
x, y = SpatialCoordinate(mesh)
f = Function(DG).interpolate(10*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

g = Function(DG).interpolate(exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 120))
h = Function(DG).interpolate(cos(x)*sin(y)+2)
k = (abs(g)*g**2)/h

a = (dot(sigma, tau) + div(tau)*k*u + dot(tau, u*grad(k)) + div(sigma)*v)*dx
L = - f*v*dx

p = point_solve(lambda x, y, z: z*x - abs(y)*y**2, function_space=DG, solver_params={'x0': (abs(g)*g**2)/h})
ps = p(g, h)

sigma2, u2 = TrialFunctions(W)
a2 = (dot(sigma2, tau) + div(tau)*ps*u2 + dot(tau, u2*grad(ps)) + div(sigma2)*v)*dx

bc0 = DirichletBC(W.sub(0), as_vector([0.0, -sin(5*x)]), 1)
bc1 = DirichletBC(W.sub(0), as_vector([0.0, sin(5*y)]), 2)

w = Function(W)
w2 = Function(W)

solve(a == L, w, bcs=[bc0, bc1])
solve(a2 == L, w2, bcs=[bc0, bc1])


sigma, u = w.split()
sigma2, u2 = w2.split()


File("poisson_mixed.pvd").write(u)

erru = norm(u-u2, norm_type='L2')
errsigma = norm(sigma-sigma2, norm_type='Hdiv')

print("\n Error u : ", erru, "\n Error sigma : ", errsigma)

plot(u)
plt.show()
