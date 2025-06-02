from firedrake.interpolation import interpolate as interpolate_default
from warnings import deprecated

@deprecated("""The symbolic `interpolate` has been moved into `firedrake.interpolation` 
            and is now the default method for interpolating in Firedrake. The need to 
            `from firedrake.__future__ import interpolate` is now unnecessary.""")
def interpolate(expr, V, *args, **kwargs):
    return interpolate_default(expr, V, *args, **kwargs)