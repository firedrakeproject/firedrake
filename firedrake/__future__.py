from ufl.domain import as_domain, extract_unique_domain
from ufl.algorithms import extract_arguments
from firedrake.mesh import VertexOnlyMeshTopology
from firedrake.interpolation import (interpolate as interpolate_old,
                                     Interpolator as InterpolatorOld,
                                     SameMeshInterpolator as SameMeshInterpolatorOld,
                                     CrossMeshInterpolator as CrossMeshInterpolatorOld)
from firedrake.cofunction import Cofunction
from functools import wraps


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


@wraps(interpolate_old)
def interpolate(expr, V, *args, **kwargs):
    default_missing_val = kwargs.pop("default_missing_val", None)
    if isinstance(V, Cofunction):
        transpose = bool(extract_arguments(expr))
        return Interpolator(
            expr, V.function_space().dual(), *args, **kwargs
        ).interpolate(V, transpose=transpose, default_missing_val=default_missing_val)
    return Interpolator(expr, V, *args, **kwargs).interpolate(default_missing_val=default_missing_val)
