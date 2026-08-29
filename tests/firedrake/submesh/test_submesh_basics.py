import os
import pytest
import numpy as np
from firedrake import *
from petsc4py import PETSc


cwd = os.path.abspath(os.path.dirname(__file__))


def test_submesh_parent():
    mesh = UnitIntervalMesh(2)

    M = FunctionSpace(mesh, "DG", 0)
    m = Function(M)
    m.dat.data[0] = 1

    cell_marker = 100
    parent = RelabeledMesh(mesh, [m], [cell_marker])

    submesh = Submesh(parent, parent.topological_dimension, cell_marker)
    assert submesh.topology.submesh_parent is parent.topology
    assert submesh.submesh_parent is parent


def test_submesh_redistribute_codim():
    # The entities of a submesh of non-zero co-dimension are not the entities
    # of its parent, so they can not inherit its orientations.
    mesh = UnitSquareMesh(2, 2)
    with pytest.raises(NotImplementedError):
        Submesh(mesh, subdomain_id="on_boundary", redistribute=True)


def _curved_mesh(nx=4, degree=2):
    """Build a unit square whose curved coordinates the plex can not carry."""
    mesh = UnitSquareMesh(nx, nx)
    V = VectorFunctionSpace(mesh, "CG", degree)
    x, y = SpatialCoordinate(mesh)
    coordinates = Function(V).interpolate(as_vector([x + 0.15 * sin(pi * y) * x * (1 - x), y]))
    return Mesh(coordinates)


@pytest.mark.parallel([1, 2, 3])
def test_submesh_curved_coordinates():
    # The plex carries no curved coordinates, so a submesh of the same
    # dimension must take them from its parent.
    mesh = _curved_mesh()
    x, _ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    mesh.mark_entities(Function(DG0).interpolate(conditional(x > 0.5, 1, 0)), 77)

    submesh = Submesh(mesh, mesh.topological_dimension, 77)
    assert submesh.coordinates.ufl_element() == mesh.coordinates.ufl_element()
    area = assemble(Constant(1.0) * dx(submesh))
    assert np.isclose(area, assemble(conditional(x > 0.5, 1.0, 0.0) * dx(mesh)))
    # The same region of an affine parent would measure exactly one half.
    assert not np.isclose(area, 0.5)


@pytest.mark.parallel([1, 2, 3])
def test_submesh_affine_coordinates():
    # An affine parent and a submesh of lower dimension carry the same element
    # up to their cell. The plex therefore already holds the right coordinates.
    mesh = UnitSquareMesh(3, 3)
    submesh = Submesh(mesh, 1, "on_boundary", label_name="exterior_facets")
    assert submesh.coordinates.ufl_element() == \
        mesh.coordinates.ufl_element().reconstruct(cell=submesh.ufl_cell())
    assert np.isclose(assemble(Constant(1.0) * dx(submesh)), 4.0)


@pytest.mark.parallel([1, 2, 3])
def test_submesh_curved_codim():
    # A parent and a submesh of lower dimension share only some of the nodes
    # on a parent cell. The cell node maps can not select those nodes.
    mesh = _curved_mesh()
    with pytest.raises(NotImplementedError):
        Submesh(mesh, 1, "on_boundary", label_name="exterior_facets")


@pytest.mark.parallel([1, 2])
def test_submesh_multiple_cell_types_coordinates():
    # A mesh with several cell types carries no coordinate Function, so a
    # submesh of it takes the coordinates the plex holds.
    mesh = Mesh(os.path.join(cwd, "..", "meshes", "mixed_cell_unit_square.msh"))
    submesh = Submesh(mesh, mesh.topological_dimension,
                      PETSc.DM.PolytopeType.TRIANGLE, label_name="celltype")
    assert submesh.ufl_cell().cellname == "triangle"
