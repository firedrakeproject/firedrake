import numpy as np
import pytest

from firedrake import *


def test_python_parloop():

    m = UnitSquareMesh(4, 4)
    fs = FunctionSpace(m, "CG", 2)
    f = Function(fs)

    class MyExpression(Expression):
        def eval(self, value, X):

            value[:] = np.dot(X, X)

    f.interpolate(MyExpression())
    X = m.coordinates
    assert assemble((f-dot(X, X))**2*dx)**.5 < 1.e-15


def test_python_parloop_vector():

    m = UnitSquareMesh(4, 4)
    fs = VectorFunctionSpace(m, "CG", 1)
    f = Function(fs)

    class MyExpression(Expression):
        def eval(self, value, X):

            value[:] = X

        def value_shape(self):
            return (2,)

    f.interpolate(MyExpression())
    X = m.coordinates
    assert assemble((f - X)**2*dx)**.5 < 1.e-15

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
