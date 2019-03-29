import pytest
from firedrake import *


@pytest.mark.parametrize(('quadrilateral'), [False, True])
@pytest.mark.parametrize(('degree'), range(4))
@pytest.mark.parametrize('direction', ['left', 'right', 'up', 'down'])
def test_dg_advection(degree, quadrilateral, direction):
    m = UnitSquareMesh(10, 10, quadrilateral=quadrilateral)
    V = FunctionSpace(m, "DG", degree)
    t = Function(V)
    v = TestFunction(V)

    d = {'left': ((1, 0), 1),
         'right': ((-1, 0), 2),
         'up': ((0, 1), 3),
         'down': ((0, -1), 4)}

    vec, bc_domain = d[direction]
    u = Constant(vec)

    def upw(t, un):
        return t('+')*un('+') - t('-')*un('-')

    n = FacetNormal(m)

    un = 0.5 * (dot(u, n) + abs(dot(u, n)))

    F = -dot(grad(v), u) * t * dx + dot(jump(v), upw(t, un)) * dS + dot(v, un*t) * ds

    bc = DirichletBC(V, Constant(1.), bc_domain, method="geometric")

    solve(F == 0, t, bcs=bc)

    assert errornorm(Constant(1.0), t) < 1.e-12
