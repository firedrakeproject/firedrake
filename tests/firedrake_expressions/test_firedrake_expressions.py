import firedrake as fd
import numpy as np
import ufl
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

test = lambda expr, x: (expr, x, np.all(fd.assemble(expr).dat.data == x))

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

tests = [
    test(one + two, 3),
    test(ufl.ln(one), 0),
    test(two ** minusthree, 0.125),
    test(ufl.sign(minusthree), -1),
    test(one + two / two ** minusthree, 17),
    assigntest(f, one + two, 3),
    iaddtest(f, two, 5),
    iaddtest(f, 2, 7),
    isubtest(f, 2, 5),
    imultest(f, 2, 10),
    idivtest(f, 2, 5),
    test(vone + vtwo, 3),
    test(ufl.ln(vone), 0),
    test(vtwo ** vminusthree, 0.125),
    test(ufl.sign(vminusthree), -1),
    test(vone + vtwo / vtwo ** vminusthree, 17),
    assigntest(vf, vone + vtwo, 3),
    iaddtest(vf, vtwo, 5),
    iaddtest(vf, 2, 7),
    isubtest(vf, 2, 5),
    imultest(vf, 2, 10),
    idivtest(vf, 2, 5)
]

exceptions = [False] * 2
try:
    vf.assign(f)
except ValueError:
    exceptions[0] = True
try:
    f.assign(vf)
except ValueError:
    exceptions[1] = True


def print_tests(tests):
    for t in tests:
        print "%s == %s : %s" % tuple(map(str, t))
    for i, t in enumerate(exceptions):
        print "Exception %d caught: %s" % (i, t)

if __name__ == "__main__":
    print_tests(tests)
