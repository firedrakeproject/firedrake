from firedrake import *
m = UnitSquareMesh(1, 1)
fs = FunctionSpace(m, "CG", 1)
f = Function(fs)

f.interpolate(Expression("x[0]"))


def external_integral():

    return assemble(f * ds)


def internal_integral():

    return assemble(f('+') * dS)

if __name__ == "__main__":
    print external_integral()
    print internal_integral()
