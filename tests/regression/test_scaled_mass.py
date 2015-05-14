import pytest
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(10, 10)


@pytest.mark.parametrize('expr',
                         ['f',
                          '2*f',
                          'tanh(f)',
                          '2 * tanh(f)',
                          'f + tanh(f)',
                          'cos(f) + sin(f)',
                          'cos(f)*cos(f) + sin(f)*sin(f)',
                          'tanh(f) + cos(f) + sin(f)',
                          '1.0/tanh(f) + 1.0/f',
                          'sqrt(f*f)',
                          '1.0/tanh(sqrt(f*f)) + 1.0/f + sqrt(f*f)'],
                         ids=lambda x: 'expr=(%s)' % x)
@pytest.mark.parametrize('value',
                         [1, 10, 20, -1, -10, -20],
                         ids=lambda x: 'f=(%d)' % x)
@pytest.mark.parametrize('typ',
                         ['Function', 'Constant'])
@pytest.mark.parametrize('vector',
                         [False, True],
                         ids=['scalar', 'vector'])
def test_math_functions(mesh, expr, value, typ, vector):
    if typ == 'Function':
        if vector:
            V = VectorFunctionSpace(mesh, 'CG', 1)
        else:
            V = FunctionSpace(mesh, 'CG', 1)
        f = Function(V)
        f.assign(value)
        if vector:
            f = dot(f, f)
    elif typ == 'Constant':
        if vector:
            f = Constant([value, value], domain=mesh)
            f = dot(f, f)
        else:
            f = Constant(value, domain=mesh)

    H = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(H)
    v = TestFunction(H)

    C = eval(expr)

    a = (C)*u*v*dx
    L = (C)*v*dx
    actual = Function(H)
    solve(a == L, actual)

    assert norm(assemble(actual - 1)) < 1e-6


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
