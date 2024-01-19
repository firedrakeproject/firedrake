import firedrake


def isconstant(expr):
    """Check whether expression is constant type.
    In firedrake this is a function in the real space
    Ie: `firedrake.Function(FunctionSpace(mesh, "R"))`"""
    if isinstance(expr, firedrake.Constant):
        raise ValueError("Firedrake Constant requires a domain to work with pyadjoint")
    return isinstance(expr, firedrake.Function) and expr.ufl_element().family() == "Real"
