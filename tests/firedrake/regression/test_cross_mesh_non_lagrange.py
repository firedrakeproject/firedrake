from firedrake import *
import pytest
import numpy as np
from finat.quadrature import QuadratureRule

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
    return VectorFunctionSpace(V.mesh(), element)


@pytest.fixture
def mesh1():
    return UnitSquareMesh(5, 5)

@pytest.fixture
def mesh2():
    return UnitSquareMesh(3, 3)

@pytest.mark.parametrize("element,degree", [("RT", 1), ("RT", 2), ("RT", 3),
                                            ("BDM", 1), ("BDM", 2), ("BDM", 3),
                                            ("BDFM", 2)])
def test_hdiv_cross_mesh_oneform(mesh1, mesh2, element, degree):
    V_source = VectorFunctionSpace(mesh1, "CG", 2)
    V_target = FunctionSpace(mesh2, element, degree)

    x, y = SpatialCoordinate(mesh1)
    f_source = Function(V_source).interpolate(as_vector([x, y]))

    # Make intermediate Quadrature space on target mesh
    Q_target = make_quadrature_space(V_target)

    # Interp V_source -> Q
    I1 = interpolate(f_source, Q_target)
    f_quadrature = assemble(I1)

    # Interp Q -> V_target
    I2 = interpolate(f_quadrature, V_target)
    f_target = assemble(I2)

    x1, y1 = SpatialCoordinate(mesh2)
    f_direct = Function(V_target).interpolate(as_vector([x1, y1]))

    assert np.allclose(f_target.dat.data_ro, f_direct.dat.data_ro)

@pytest.mark.parametrize("element,degree", [("N1curl", 1), ("N1curl", 2), ("N1curl", 3),
                                            ("N2curl", 1), ("N2curl", 2), ("N2curl", 3),])
def test_hcurl_cross_mesh_oneform(mesh1, mesh2, element, degree):
    V_source = VectorFunctionSpace(mesh1, "CG", 2)
    V_target = FunctionSpace(mesh2, element, degree)

    x, y = SpatialCoordinate(mesh1)
    f_source = Function(V_source).interpolate(as_vector([x, y]))

    # Make intermediate Quadrature space on target mesh
    Q_target = make_quadrature_space(V_target)

    # Interp V_source -> Q
    I1 = interpolate(f_source, Q_target)
    f_quadrature = assemble(I1)

    # Interp Q -> V_target
    I2 = interpolate(f_quadrature, V_target)
    f_target = assemble(I2)

    x1, y1 = SpatialCoordinate(mesh2)
    f_direct = Function(V_target).interpolate(as_vector([x1, y1]))

    assert np.allclose(f_target.dat.data_ro, f_direct.dat.data_ro)
