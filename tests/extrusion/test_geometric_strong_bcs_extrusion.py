import pytest
from firedrake import *


@pytest.mark.parametrize(('degree'), range(4))
@pytest.mark.parametrize('direction', ['left', 'right', 'up', 'down'])
@pytest.mark.parametrize('layers', [1, 10])
def test_dg_advection(degree, direction, layers):
    m = UnitIntervalMesh(10)
    m = ExtrudedMesh(m, layers=layers)
    V = FunctionSpace(m, "DG", degree)
    t = Function(V)
    v = TestFunction(V)

    d = {'left': ((1, 0), 1),
         'right': ((-1, 0), 2),
         'up': ((0, 1), 'bottom'),
         'down': ((0, -1), 'top')}

    vec, bc_domain = d[direction]
    u = Constant(vec)

    def upw(t, un):
        return t('+')*un('+') - t('-')*un('-')

    n = FacetNormal(m)

    un = 0.5 * (dot(u, n) + abs(dot(u, n)))

    F = -dot(grad(v), u) * t * dx + dot(jump(v), upw(t, un)) * (dS_v + dS_h) + dot(v, un*t) * (ds_tb + ds_v)

    bc = DirichletBC(V, 1., bc_domain, method="geometric")

    solve(F == 0, t, bcs=bc)

    assert errornorm(Constant(1.0), t) < 1.e-12
