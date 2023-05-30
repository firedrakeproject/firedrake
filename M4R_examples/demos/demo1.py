
from firedrake import *
from firedrake.petsc import PETSc
try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)



Lx   = 1.
Ly   = 1.
n0   = 50
mesh = RectangleMesh(n0, n0, Lx, Ly, reorder=None)


Vcg  = FunctionSpace(mesh,'CG',3)
bc = DirichletBC(Vcg, 0.0, "on_boundary")
beta = Constant('1.0')
F    = Constant('1.0')
phi, psi = TestFunction(Vcg), TrialFunction(Vcg)
a =  beta*phi*psi.dx(0)*dx
eigenprob = LinearEigenproblem(a, bcs=bc)
num_eigenvalues = 1
eigensolver = LinearEigensolver(eigenprob, num_eigenvalues)
eigensolver.solve()
lam = eigensolver.eigenvalue(0)
eigenmodes_real, eigenmodes_imag = eigensolver.eigenfunction(0)
print("Leading eigenvalue is:", lam)




# try:
#     import matplotlib.pyplot as plt
#     fig, axes = plt.subplots()
#     colors = tripcolor(eigenmodes_real, axes=axes)
#     fig.colorbar(colors)

#     fig, axes = plt.subplots()
#     colors = tripcolor(eigenmodes_imag, axes=axes)
#     fig.colorbar(colors)
# except ImportError:
#     warning("Matplotlib not available, not plotting eigemodes")
