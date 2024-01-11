from firedrake import *

mesh = UnitSquareMesh(1, 1)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)

u = TrialFunction(V)
v = TestFunction(V)

original_form = u * v * dx
matrix = assemble(u * v * dx)
print(matrix.M.values)  # getting the 4x4 matrix values to compare with the ones later

bc = DirichletBC(V, 0, 4)
V_res = RestrictedFunctionSpace(V, name="Restricted", bcs=[bc])

u2 = TrialFunction(V_res)
v2 = TestFunction(V_res)
restricted_form = u2 * v2 * dx

matrix_res = assemble(u2 * v2 * dx)  # it works!
print(matrix_res.M.values)

matrix_normal_bcs = assemble(u * v * dx, bcs=[bc])
print(matrix_normal_bcs.M.values)
