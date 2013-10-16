import pytest

import numpy as np
import ufl

import firedrake as fd

mesh = fd.UnitSquareMesh(5, 5)

fs = fd.FunctionSpace(mesh, "Lagrange", 1)
vfs = fd.VectorFunctionSpace(mesh, "Lagrange", 1)
f = fd.Function(fs, name="f")
one = fd.Function(fs, name="one")
two = fd.Function(fs, name="two")
minusthree = fd.Function(fs, name="minusthree")

vf = fd.Function(vfs, name="vf")
vone = fd.Function(vfs, name="vone")
vtwo = fd.Function(vfs, name="vtwo")
vminusthree = fd.Function(vfs, name="vminusthree")

two.interpolate(fd.Expression("2"))
one.interpolate(fd.Expression("1"))
minusthree.interpolate(fd.Expression("-3"))

vone.assign(1)
vtwo.assign(2)
vminusthree.assign(-3)

exprtest = lambda expr, x: (expr, x, np.all(fd.assemble(expr).dat.data == x))

assigntest = lambda f, expr, x: (str(f) + " = " + str(expr) + ", " + str(f), x,
                                 np.all(f.assign(expr).dat.data == x))


def iaddtest(f, expr, x):
    f += expr
    return (str(f) + " += " + str(expr) + ", " + str(f), x,
            np.all(f.dat.data == x))


def isubtest(f, expr, x):
    f -= expr
    return (str(f) + " -= " + str(expr) + ", " + str(f), x,
            np.all(f.dat.data == x))


def imultest(f, expr, x):
    f *= expr
    return (str(f) + " *= " + str(expr) + ", " + str(f), x,
            np.all(f.dat.data == x))


def idivtest(f, expr, x):
    f /= expr
    return (str(f) + " /= " + str(expr) + ", " + str(f), x,
            np.all(f.dat.data == x))


@pytest.mark.parametrize("test", [
    exprtest(one + two, 3),
    exprtest(ufl.ln(one), 0),
    exprtest(two ** minusthree, 0.125),
    exprtest(ufl.sign(minusthree), -1),
    exprtest(one + two / two ** minusthree, 17),
    assigntest(f, one + two, 3),
    iaddtest(f, two, 5),
    iaddtest(f, 2, 7),
    isubtest(f, 2, 5),
    imultest(f, 2, 10),
    idivtest(f, 2, 5),
    exprtest(vone + vtwo, 3),
    exprtest(ufl.ln(vone), 0),
    exprtest(vtwo ** vminusthree, 0.125),
    exprtest(ufl.sign(vminusthree), -1),
    exprtest(vone + vtwo / vtwo ** vminusthree, 17),
    assigntest(vf, vone + vtwo, 3),
    iaddtest(vf, vtwo, 5),
    iaddtest(vf, 2, 7),
    isubtest(vf, 2, 5),
    imultest(vf, 2, 10),
    idivtest(vf, 2, 5)
])
def test_expressions(test):
    assert test[2]


@pytest.mark.parametrize(("f1", "f2"), [(vf, f), (f, vf)])
def test_exceptions(f1, f2):
    with pytest.raises(ValueError):
        f1.assign(f2)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
