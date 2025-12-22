from firedrake import *
import pytest
import numpy as np
from finat.quadrature import QuadratureRule
from functools import partial


def fs_shape(V):
    shape = V.ufl_function_space().value_shape
    if len(shape) == 1:
        fs_type = partial(VectorFunctionSpace, dim=shape[0])
    elif len(shape) == 2:
        fs_type = partial(TensorFunctionSpace, shape=shape)
    else:
        raise ValueError("Invalid function space shape")
    return fs_type

def make_quadrature_space(V):
    """Builds a quadrature space on the target mesh.
    
    This space has point evaluation dofs at the quadrature PointSet
    of the target space's element

    """
    fe = V.finat_element
    _, ps = fe.dual_basis
    wts = np.full(len(ps.points), np.nan)  # These can be any number since we never integrate
    scheme = QuadratureRule(ps, wts, ref_el=fe.cell)
    element = FiniteElement("Quadrature", degree=fe.degree, quad_scheme=scheme)
    return fs_shape(V)(V.mesh(), element)


@pytest.fixture(params=[("RT", 1), ("RT", 2), ("RT", 3), ("BDM", 1), ("BDM", 2), ("BDM", 3),
                        ("BDFM", 2), ("HHJ", 2),("N1curl", 1), ("N1curl", 2), ("N1curl", 3),
                        ("N2curl", 1), ("N2curl", 2), ("N2curl", 3), ("GLS", 1), ("GLS", 2),
                        ("GLS", 3), ("GLS2", 1), ("GLS2", 2), ("GLS2", 3)],
                        ids=lambda x: f"{x[0]}_{x[1]}")
def V_target(request):
    element, degree = request.param
    mesh = UnitSquareMesh(3, 3)
    return FunctionSpace(mesh, element, degree)


def test_cross_mesh_oneform(V_target):
    mesh1 = UnitSquareMesh(5, 5)
    mesh2 = V_target.mesh()
    x, y = SpatialCoordinate(mesh1)
    x1, y1 = SpatialCoordinate(mesh2)

    shape = V_target.ufl_function_space().value_shape
    if len(shape) == 1:
        fs_type = partial(VectorFunctionSpace, dim=shape[0])
        expr1 = as_vector([x, y])
        expr2 = as_vector([x1, y1])
    elif len(shape) == 2:
        fs_type = partial(TensorFunctionSpace, shape=shape)
        expr1 = as_tensor([[x, x*y], [x*y, y]])
        expr2 = as_tensor([[x1, x1*y1], [x1*y1, y1]])
    else:
        raise ValueError("Unsupported target space shape")
    
    V_source = fs_type(mesh1, "CG", 2)
    f_source = Function(V_source).interpolate(expr1)

    # Make intermediate Quadrature space on target mesh
    Q_target = make_quadrature_space(V_target)

    # Interp V_source -> Q
    I1 = interpolate(f_source, Q_target)
    f_quadrature = assemble(I1)

    # Interp Q -> V_target
    I2 = interpolate(f_quadrature, V_target)
    f_target = assemble(I2)

    f_direct = Function(V_target).interpolate(expr2)

    assert np.allclose(f_target.dat.data_ro, f_direct.dat.data_ro)
