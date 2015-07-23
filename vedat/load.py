import numpy as np
from ctypes import POINTER, c_double

from firedrake import *

from c_evaluate import c_evaluate as make_c_evaluate
from cfunction import cFunction

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "P", 2)
f = Function(V).interpolate(Expression("x[0]*(1.0 - x[0]) + 0.5*x[1]"))

c_evaluate = make_c_evaluate(f)

x = np.array([0.2, 0.65], dtype=float)


def evaluate(x):
    result = np.zeros(1, dtype=float)
    retval = c_evaluate(cFunction(f),
                        x.ctypes.data_as(POINTER(c_double)),
                        result.ctypes.data_as(POINTER(c_double)))
    assert retval != -1, "Point is not in the domain."
    return result


def analytic(x):
    return np.array([x[0]*(1.0 - x[0]) + 0.5*x[1]])


Lx = 1.0
Ly = 1.0
nx = 64
ny = 64
dx = float(Lx) / nx
dy = float(Ly) / ny
xcoords = np.arange(0.0, Lx + 0.01 * dx, dx)
ycoords = np.arange(0.0, Ly + 0.01 * dy, dy)
coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)

for p in coords:
    assert np.allclose(evaluate(p), analytic(p))
