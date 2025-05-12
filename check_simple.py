from firedrake import *


mesh = UnitSquareMesh(2, 2)
x, _ = SpatialCoordinate(mesh)

assert np.allclose(assemble(1 * dx(domain=mesh)), 1)
assert np.allclose(assemble(x * dx(domain=mesh)), 0.5)

plex = mesh.topology_dm
coords_vec = plex.getCoordinatesLocal()
coords_vec *= 2

assert np.allclose(assemble(1 * dx(domain=mesh)), 4)
assert np.allclose(assemble(x * dx(domain=mesh)), 4)

print("Success! Yay!")
