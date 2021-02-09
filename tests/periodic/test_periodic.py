from firedrake import *
import matplotlib.pyplot as plt

coarse = False
if coarse:
    mesh = Mesh("geom/p2d-coarse.msh",reorder=False, periodic=True)
else:
    mesh = Mesh("geom/p2d.msh",reorder=False, periodic=True)

#mesh = PeriodicRectangleMesh(2,2,0.6,0.5, direction="x")
Vd = VectorFunctionSpace(mesh,"DG",1)
Vc = VectorFunctionSpace(mesh,"CG",1)
u = Function(Vc).interpolate(SpatialCoordinate(mesh))
ud = Function(Vd).interpolate(SpatialCoordinate(mesh))
File("output/results.pvd").write(u,ud)

V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
v = TestFunction(V)

a = inner(grad(u),grad(v))*dx
l = inner(Constant(1e-1),v)*dx

bc = DirichletBC(V, Constant(0), [1,3])
#bc = DirichletBC(V, Constant(0),[1,2])

if coarse:
    solve(a-l==0, u, bc, nullspace=VectorSpaceBasis(constant=True), transpose_nullspace=VectorSpaceBasis(constant=True), solver_parameters={"snes_monitor":None})
else:
    # Good Convergence!
    solve(a-l==0, u, bc, solver_parameters={"snes_monitor":None})

File("output/lapl.pvd").write(u)
tricontourf(u)
plt.show()
