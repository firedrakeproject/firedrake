from firedrake.interpolation import interpolate, Interpolator
from firedrake.cofunction import Cofunction


__all__ = ("interpolate",)


# Monkey patch interpolation
Interpolator.interpolate = Interpolator._interpolate_future
Cofunction.interpolate = Cofunction._interpolate_future
