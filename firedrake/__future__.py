from firedrake.interpolation import interpolate
from functools import wraps

def interpolate(expr, V, *args, **kwargs):
    return interpolate(expr, V, *args, **kwargs)