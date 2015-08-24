import ctypes
from os import path

from cfunction import cFunction, c_evaluate


def simple_d2s(f, size_u, size_v, steps):
    p_cf = cFunction(f)
    init_file = path.join(path.dirname(__file__), 'simple_d2s_init.o')
    diderot_file = path.join(path.dirname(__file__), 'simple_d2s.o')
    call = c_evaluate(f, "callDiderot2_step", ldargs=[init_file, diderot_file, "-lteem"])

    return call(p_cf, size_u, size_v, ctypes.c_float(steps))

if __name__ == "__main__":
    from firedrake import *
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression("sin(2*pi *(x[0]-x[1]))"))
    print simple_d2s(f, 100, 100, 0.01)
