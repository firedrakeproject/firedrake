from firedrake import *

def test_high_order_location():
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 3, variant="equispaced")
    f = Function(V)
    f.interpolate(mesh.coordinates)

    def warp(x):
        return 5.0*x*(2*x - 1)

    warp_indices = np.where((f.dat.data[:, 0] > 0.0) & (f.dat.data[:, 0] < 0.5) & (f.dat.data[:, 1] == 0.0))[0]
    f.dat.data[warp_indices, 1] = warp(f.dat.data[warp_indices, 0])

    mesh = Mesh(f)
    
    # The point (0.25, -0.6) *is* in the mesh, but falls outside the Lagrange bounding box
    # The below used to return (None, None), but projecting to Bernstein coordinates
    # allows us to locate the cell.
    assert mesh.locate_cell([0.25, -0.6], tolerance=0.001) is not None
    # The point (0.25, -0.7) is outside the mesh, but inside the Bernstein bounding box.
    # This should return (None, None).
    assert mesh.locate_cell([0.25, -0.7], tolerance=0.001) is None