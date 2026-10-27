import numpy as np
from firedrake import *


mesh = UnitSquareMesh(3, 3)

def demo1():
    V = VectorFunctionSpace(mesh, "P", 1)

    f = Function(V)
    g = Function(V)

    f.dat.assign(2*g.dat+3, eager=True, eager_strategy="compile", compiler_parameters={"backend": "gem"})


def demo2():
    V = FunctionSpace(mesh, "P", 2)

    c = Constant(10)
    v = TestFunction(V)

    # do a regular one first to make sure that it still runs
    # assemble(c*v*dx)

    gem_a = assemble(c*v*dx, pyop3_compiler_parameters={"backend": "gem"}, form_compiler_parameters={"backend": "gem"})


def demo3():
    V = FunctionSpace(mesh, "P", 2)

    x, y = SpatialCoordinate(mesh)
    v = TestFunction(V)

    gem_b = assemble(
        x*y*v*dx,
        pyop3_compiler_parameters={"backend": "gem"},
        form_compiler_parameters={"backend": "gem"},
    )

    # The P2 basis functions sum to 1, so the entries should sum to the integral of x*y
    # over the unit square - 1/4.
    total = gem_b.dat.data_ro.sum()
    print(f"sum of entries = {total!r}, expected 0.25")
    assert np.isclose(total, 0.25)

demo1()
# demo2()
# demo3()
