"""
Tests two general spherical meshes: that they can be formed without error and
that they have the correct number of cells/edges/vertices.
"""

from firedrake import (dx, Function, FunctionSpace, assemble, pi,
                       IcosahedralSphereMesh, CubedSphereMesh)
import pytest


@pytest.mark.parametrize("mesh_type", ["icosahedral", "cubed_sphere"])
@pytest.mark.parametrize("num", [1, 3, 7])
def test_build_mesh(mesh_type, num):

    # n is the number of cells per edge of panel

    radius = 15.5
    sphere_area = 4*pi*radius**2

    if mesh_type == "icosahedral":
        mesh = IcosahedralSphereMesh(radius, num_cells_per_edge_of_panel=num, degree=2)
        ncells = 20*num**2
        nedges = 30*num + 20*3*(num-1)*num/2
        if num > 1:
            nvertices = 12 + 30*(num-1) + 20*(num-2)*(num-1)/2
        else:
            nvertices = 12
    elif mesh_type == "cubed_sphere":
        mesh = CubedSphereMesh(radius, num_cells_per_edge_of_panel=num, degree=2)
        ncells = 6*num**2
        nedges = 12*num + 6*2*num*(num-1)
        if num > 1:
            nvertices = 8 + 12*(num-1) + 6*(num-1)**2
        else:
            nvertices = 8
    else:
        raise ValueError(f'mesh type {mesh_type} not recognised')

    # Check that mesh has correct number of cells/edges/vertices
    assert mesh.num_cells() == ncells, \
        f'number of cells for {mesh_type} mesh appears to be incorrect'
    assert mesh.num_edges() == nedges, \
        f'number of edges for {mesh_type} mesh appears to be incorrect'
    assert mesh.num_vertices() == nvertices, \
        f'number of vertices for {mesh_type} mesh appears to be incorrect'

    # Check surface area is roughly correct
    V = FunctionSpace(mesh, "DG", 0)
    ones = Function(V).assign(1.0)
    area = assemble(ones * dx)
    assert abs(area - sphere_area) / sphere_area < 0.02, \
        f'area of {mesh_type} mesh appears to be incorrect'
