from firedrake import *
from firedrake.adjoint import *
from numpy.testing import assert_allclose
continue_annotation()
my_ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size)

mesh = UnitSquareMesh(1, 1)
R = FunctionSpace(mesh, "R", 0)

n = 3
x = [Function(R) for i in range(n)]
c = [Control(xi) for xi in x]

# Rosenbrock function https://en.wikipedia.org/wiki/Rosenbrock_function
# with minimum at x = (1, 1, 1, ...)
f = sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(n-1))

J = assemble(f * dx(domain=mesh))
rf = EnsembleReducedFunctional(J, c, my_ensemble)
result = minimize(rf)
assert_allclose([float(xi) for xi in result], 1., rtol=1e-4)