import pytest
import numpy as np
from firedrake import *


def test_submesh_subdomain_id_union():
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    M = FunctionSpace(mesh, "DG", 0)
    m1 = Function(M).interpolate(conditional(lt(x, 0.5), 1, 0))
    m2 = Function(M).interpolate(conditional(lt(y, 0.5), 1, 0))
    mesh.mark_entities(m1, 111)
    mesh.mark_entities(m2, 222)

    subdomain_id = [111, 222]
    submesh1 = Submesh(mesh, mesh.topological_dimension, subdomain_id=subdomain_id)

    m3 = Function(M).interpolate(m1 + m2 - m1 * m2)
    expected = assemble(m3*dx)
    assert abs(assemble(1*dx(domain=submesh1)) - expected) < 1E-12

    mesh.mark_entities(m3, 333)
    submesh2 = Submesh(mesh, mesh.topological_dimension, 333)
    assert submesh2.cell_set.size == submesh1.cell_set.size
    assert np.allclose(submesh2.coordinates.dat.data, submesh1.coordinates.dat.data)


@pytest.mark.parametrize("subdomain_id", ["on_boundary", (1, 3, 6)])
def test_submesh_facet_subdomain_id_union(subdomain_id):
    mesh = UnitCubeMesh(2, 2, 2)
    submesh1 = Submesh(mesh, mesh.topological_dimension - 1, subdomain_id=subdomain_id)
    if subdomain_id == "on_boundary":
        area = assemble(1*ds(domain=mesh))
    else:
        area = assemble(1*ds(subdomain_id, domain=mesh))
    assert abs(assemble(1*dx(domain=submesh1)) - area) < 1E-12

    DGT = FunctionSpace(mesh, "DGT", 0)
    facet_function = Function(DGT)
    DirichletBC(DGT, 1, subdomain_id).apply(facet_function)
    facet_value = 999
    rmesh = RelabeledMesh(mesh, [facet_function], [facet_value])
    submesh2 = Submesh(rmesh, mesh.topological_dimension - 1, facet_value)
    assert submesh2.cell_set.size == submesh1.cell_set.size
    assert np.allclose(submesh2.coordinates.dat.data, submesh1.coordinates.dat.data)
