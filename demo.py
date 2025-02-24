from firedrake import *


# plex = PETSc.DMPlex().createBoxMesh(
#     (2, 2, 2),
#     lower=(0, 0, 0),
#     upper=(1, 1, 1),
#     simplex=False,
#     periodic=True,
#     interpolate=True,
#     sparseLocalize=False,
# )
# mesh = Mesh(plex)

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "DG", 1)
x, _ = SpatialCoordinate(mesh)

assert np.allclose(assemble(1 * dx(domain=mesh)), 1)
assert np.allclose(assemble(x * dx(domain=mesh)), 0.5)

plex = mesh.topology_dm
coords_vec = plex.getCoordinatesLocal()
coords_vec *= 2

assert np.allclose(assemble(1 * dx(domain=mesh)), 4)
assert np.allclose(assemble(x * dx(domain=mesh)), 4)

print("Success! Yay!")
