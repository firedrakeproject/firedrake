from firedrake import *
import pyop3 as op3
import numpy as np, cupy as cp

mesh = UnitSquareMesh(3,3)
V = FunctionSpace(mesh, "CG", 1)
f = Function(V).assign(10)
g = Function(V)

gpu = op3.CUDAGPU()

g.dat.assign(2 * f.dat, eager=True, eager_strategy="compile")
print("DONE ASSIGN OPERATION\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
# with op3.offloading(gpu):
# 	g.dat.assign(2 * f.dat, eager=True, eager_strategy="compile")
# 	assert isinstance(g.dat.data_ro, cp.ndarray) # Device
# 	assert (g.dat.data_ro == 20).all()
	
assert isinstance(g.dat.data_ro, np.ndarray) # Host
assert (g.dat.data_ro == 20).all()

