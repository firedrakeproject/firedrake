from firedrake import *
import pyop3 as op3
import numpy as np

mesh = UnitSquareMesh(3,3)

V = FunctionSpace(mesh, "CG", 1)
f = Function(V).assign(10)
g = Function(V)

g.dat.assign(
    2 * f.dat, 
    eager=True, 
    eager_strategy="compile", 
    compiler_parameters={"codegen": "mlir"}
)
assert (g.dat.data_ro == 20).all()
