from firedrake import *


mesh = UnitSquareMesh(3, 3)

def demo1():
    V = VectorFunctionSpace(mesh, "P", 1)

    f = Function(V)
    g = Function(V)

    f.dat.assign(2*g.dat+3, eager=True, eager_strategy="compile", compiler_parameters={"backend": "gem"})


def demo2():
    V = FunctionSpace(mesh, "P", 1)

    c = Constant(10)
    v = TestFunction(V)

    # do a regular one first to make sure that it still runs
    assemble(c*v*dx)

    assemble(c*v*dx, pyop3_compiler_parameters={"backend": "gem"}, form_compiler_parameters={"backend": "gem"})


# demo1()
demo2()
