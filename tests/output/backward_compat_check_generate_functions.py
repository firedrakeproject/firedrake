from firedrake import *


def _get_expr(V):
    mesh = V.mesh()
    dim = mesh.geometric_dimension()
    shape = V.ufl_element().value_shape()
    if dim == 2:
        x, y = SpatialCoordinate(mesh)
        z = x * y
    elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)
    if shape == (8, ):
        return as_vector([x, y, z, z * z, x * y, y * z, z * x, x * y * z])
    else:
        raise ValueError(f"Invalid shape {shape}")


def _get_function(V, name=None):
    return Function(V, name=name).project(_get_expr(V))
