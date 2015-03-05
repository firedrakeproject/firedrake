from firedrake import *
import numpy as np


def test_set_top():
    m = UnitSquareMesh(2, 2)
    me = ExtrudedMesh(m, 2)

    fs = FunctionSpace(me, "CG", 1)

    f = Function(fs)

    f.interpolate(Expression("x[0]"))

    f_top = f.extract_subfunction("top")

    print f_top.dat.data
