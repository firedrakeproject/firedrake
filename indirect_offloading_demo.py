from firedrake import *
import pyop3 as op3
import numpy as np, cupy as cp

import pyop3.debug_flags

mesh = UnitSquareMesh(3,3)

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

pyop3.debug_flags.hit_assign = True
b = assemble(conj(v) * dx)
pyop3.debug_flags.hit_assign = False

