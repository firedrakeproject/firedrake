from firedrake.interpolation import interpolate as interpolate_default, Interpolator as Interpolator_default
import warnings


def interpolate(expr, V, *args, **kwargs):
    warnings.warn("""The symbolic `interpolate` has been moved into `firedrake`
                  and is now the default method for interpolating in Firedrake. The need to
                  `from firedrake.__future__ import interpolate` is now deprecated.""", FutureWarning)
    return interpolate_default(expr, V, *args, **kwargs)


class Interpolator(Interpolator_default):
    def __new__(cls, *args, **kwargs):
        warnings.warn("""The symbolic `Interpolator` has been moved into `firedrake`.
                      The need to `from firedrake.__future__ import Interpolator` is now deprecated.""", FutureWarning)
        return Interpolator_default(*args, **kwargs)
