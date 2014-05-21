import pytest
from firedrake import *


@pytest.mark.parametrize(('degree'), range(5))
def test_dg_advection(degree):
    m = UnitSquareMesh(10, 10)
    V = FunctionSpace(m, "DG", degree)
    V_u = VectorFunctionSpace(m, "CG", 1)
    t = Function(V)
    v = TestFunction(V)
    u = Function(V_u)

    u.assign(Constant((1, 0)))

    def upw(t, un):
        return t('+')*un('+') - t('-')*un('-')

    n = FacetNormal(m)

    un = 0.5 * (dot(u, n) + abs(dot(u, n)))

    F = -dot(grad(v), u) * t * dx + dot(jump(v), upw(t, un)) * dS + dot(v, un*t) * ds

    bc = DirichletBC(V, 1., 1, method="geometric")

    solve(F == 0, t, bcs=bc)

    assert errornorm(Constant(1.0), t) < 1.e-12
