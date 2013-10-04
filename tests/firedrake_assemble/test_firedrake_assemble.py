import firedrake as fd
mesh = fd.UnitSquareMesh(5, 5)

fs = fd.FunctionSpace(mesh, "Lagrange", 1)

f = fd.Function(fs, name="f")
one = fd.Function(fs, name="one")

uhat = fd.TrialFunction(fs)
v = fd.TestFunction(fs)

M = uhat * v * fd.dx

f.interpolate(fd.Expression("x[0]"))
one.interpolate(fd.Expression("1"))
