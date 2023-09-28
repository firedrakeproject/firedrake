from firedrake.interpolation import interpolate, Interpolator


__all__ = ("interpolate",)


# Monkey patch interpolation
Interpolator.interpolate = Interpolator._interpolate_future
