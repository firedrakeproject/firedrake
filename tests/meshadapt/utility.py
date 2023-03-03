from firedrake import *
from firedrake.meshadapt import *


def uniform_mesh(dim, n=5, recentre=False):
    """
    Create a uniform mesh of a specified dimension and size.

    :param dim: the topological dimension
    :param n: the number of subdivisions in each coordinate direction
    :param recentre: if ``True``, the mesh is re-centred on the origin
    """
    if dim == 2:
        mesh = UnitSquareMesh(n, n)
    elif dim == 3:
        mesh = UnitCubeMesh(n, n, n)
    else:
        raise ValueError(f"Can only adapt in 2D or 3D, not {dim}D")
    if recentre:
        coords = Function(mesh.coordinates)
        coords.interpolate(2 * (coords - as_vector([0.5] * dim)))
        return Mesh(coords)
    return mesh


def uniform_metric(mesh, a=100.0, metric_parameters={}):
    """
    Create a metric which is just the identity
    matrix scaled by `a` at each vertex.

    :param mesh: the mesh to define the metric upon
    :param a: the scale factor for the identity
    :param: parameters to pass to PETSc's Riemannian metric
    """
    dim = mesh.topological_dimension()
    metric = RiemannianMetric(mesh)
    metric.interpolate(a * Identity(dim))
    metric.set_parameters(metric_parameters)
    return metric
