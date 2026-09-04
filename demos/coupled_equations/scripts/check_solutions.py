from firedrake import *

n = 8
mesh1 = UnitSquareMesh(n, n, quadrilateral=True)
mesh2 = UnitSquareMesh(n, n, quadrilateral=True)
mesh2.coordinates.dat.data[:, 0] += 1.0  # Shift to the right by 1

x1, y1 = SpatialCoordinate(mesh1)
x2, y2 = SpatialCoordinate(mesh2)
#u1_exact = x1 * sin(pi * y1) ** 2
#u2_exact = sin(pi * y2) ** 2 * (x2 - (x2 - 1) ** 2 / 2)

u1_exact = (x1 * y1**2 * (1-y1)**2) + (x1 * (1-x1) * y1**3 * (1-y1)**3)
u2_exact = (y2**2 * (1-y2)**2) * (2*(x2-1)**3 - 3*(x2-1)**2 + 1)

f1 = -div(grad(u1_exact))
f2 = -div(grad(u2_exact)) + u2_exact

n1 = FacetNormal(mesh1)
n2 = FacetNormal(mesh2)
dx1 = Measure("dx", domain=mesh1)
dx2 = Measure("dx", domain=mesh2)
ds1 = Measure("ds", domain=mesh1)
ds2 = Measure("ds", domain=mesh2)

ds1_interface = ds1(2)
ds2_interface = ds2(1)

# External boundaries of mesh1
ds1_x0 = ds1(1)
ds1_y0 = ds1(3)
ds1_y1 = ds1(4)
bc_error_u1 = assemble(u1_exact * (ds1_x0 + ds1_y0 + ds1_y1))
print("Poisson Dirichlet boundary error =", bc_error_u1)

# Neumann boundary condition on mesh1
flux1 = dot(grad(u1_exact), n1)
flux2 = dot(grad(u2_exact), n2)
flux_error = assemble(flux1 * ds1_interface) + assemble(flux2 * ds2_interface)
print("Interface flux mismatch =", flux_error)

# Dirichlet boundary condition
interface_error = assemble(u1_exact * ds1_interface) - assemble(u2_exact * ds2_interface)
print("Interface Dirichlet mismatch =", interface_error)

# External boundaries of mesh2
ds2_x2 = ds2(2)
ds2_y0 = ds2(3)
ds2_y1 = ds2(4)
flux2 = dot(grad(u2_exact), n2)
bc_error_u2 = assemble(flux2 * (ds2_x2 + ds2_y0 + ds2_y1))
print("Helmholtz Neumann boundary error =", bc_error_u2)

