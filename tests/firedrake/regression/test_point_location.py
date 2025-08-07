from firedrake import *


def test_high_order_location():
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 3, variant="equispaced")
    f = Function(V)
    f.interpolate(mesh.coordinates)

    def warp(x, p):
        return p * x * (2 * x - 1)

    warp_indices = np.where((f.dat.data[:, 0] > 0.0) & (f.dat.data[:, 0] < 0.5) & (f.dat.data[:, 1] == 0.0))[0]
    f.dat.data[warp_indices, 1] = warp(f.dat.data[warp_indices, 0], 5.0)

    mesh = Mesh(f)

    # The point (0.25, -0.6) *is* in the mesh, but falls outside the Lagrange bounding box
    # The below used to return (None, None), but projecting to Bernstein coordinates
    # allows us to locate the cell.
    assert mesh.locate_cell([0.25, -0.6], tolerance=0.001) is not None
    # The point (0.25, -0.7) is outside the mesh, but inside the Bernstein bounding box.
    # This should return (None, None).
    assert mesh.locate_cell([0.25, -0.7], tolerance=0.001) is None

    # Change mesh coordinates to check that the bounding box is recalculated
    mesh.coordinates.dat.data_wo[warp_indices, 1] = warp(mesh.coordinates.dat.data_ro[warp_indices, 0], 8.0)
    assert mesh.locate_cell([0.25, -0.6], tolerance=0.0001) is not None
    assert mesh.locate_cell([0.25, -0.7], tolerance=0.0001) is not None
    assert mesh.locate_cell([0.25, -0.95], tolerance=0.0001) is not None
    assert mesh.locate_cell([0.25, -1.05], tolerance=0.0001) is None


def test_high_order_location_internal():
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 3, variant="equispaced")
    f = Function(V)
    f.interpolate(mesh.coordinates)

    warp_indices = np.where((f.dat.data[:, 0] > 0.0) & (f.dat.data[:, 0] < 0.5) & np.isclose(f.dat.data[:, 1], 0.5))[0]
    f.dat.data[warp_indices, 1] += 0.1
    mesh = Mesh(f)

    assert mesh.locate_cell([0.25, 0.605], tolerance=0.0001) == 1
    assert mesh.locate_cell([0.25, 0.62], tolerance=0.0001) == 3
