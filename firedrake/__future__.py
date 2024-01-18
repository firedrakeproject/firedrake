from ufl.domain import as_domain, extract_unique_domain
from firedrake.mesh import VertexOnlyMeshTopology
from firedrake.interpolation import (interpolate as interpolate_old,
                                     Interpolator as InterpolatorOld,
                                     SameMeshInterpolator as SameMeshInterpolatorOld,
                                     CrossMeshInterpolator as CrossMeshInterpolatorOld)


__all__ = ("interpolate", "Interpolator")


class Interpolator(InterpolatorOld):
    def __new__(cls, expr, V, **kwargs):
        target_mesh = as_domain(V)
        source_mesh = extract_unique_domain(expr) or target_mesh
        if target_mesh is not source_mesh:
            if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
                return object.__new__(SameMeshInterpolator)
            else:
                return object.__new__(CrossMeshInterpolator)
        else:
            return object.__new__(SameMeshInterpolator)

    interpolate = InterpolatorOld._interpolate_future


class SameMeshInterpolator(Interpolator, SameMeshInterpolatorOld):
    pass


class CrossMeshInterpolator(Interpolator, CrossMeshInterpolatorOld):
    pass


def interpolate(*args, **kwargs):
    default_missing_val = kwargs.pop("default_missing_val", None)
    return Interpolator(*args, **kwargs).interpolate(default_missing_val=default_missing_val)


interpolate.__doc__ = interpolate_old.__doc__
