import ctypes
from os import path

from cfunction import *

__all__ = ['simple_d2s']


def simple_d2s(f, size_u, size_v, steps):
    p_cf = cFunction(f)
    init_file = path.join(path.dirname(__file__), 'simple_d2s_init.o')
    diderot_file = path.join(path.dirname(__file__), 'simple_d2s.o')
    call = make_c_evaluate(f, "callDiderot2_step", ldargs=[init_file, diderot_file, "-lteem"])

    return call(p_cf, size_u, size_v, ctypes.c_float(steps))
