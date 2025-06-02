from firedrake.interpolation import interpolate
from functools import wraps
import warnings

def interpolate(expr, V, *args, **kwargs):
    warnings.warn("""The symbolic `interpolate` has been moved into `firedrake.interpolation` 
                  and is now the default method for interpolating in Firedrake. The need to 
                  `from firedrake.__future__ import interpolate` is now unnecessary.""", FutureWarning)
    return interpolate(expr, V, *args, **kwargs)
