from firedrake import *
from firedrake.petsc import PETSc
try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)


# We specify the geometry to be a square geometry with :math:`50` cells
# with length :math:`1`. ::

Lx   = 1.
Ly   = 1.
n0   = 50
mesh = RectangleMesh(n0, n0, Lx, Ly, reorder=None)

# Next we define the function spaces within which our solution will
# reside. ::

Vcg  = FunctionSpace(mesh,'CG',3)

# We impose zero Dirichlet boundary conditions, in a strong sense, which
# guarantee that we have no-normal flow at the boundary walls. ::

bc = DirichletBC(Vcg, 0.0, "on_boundary")

# The two non-dimensional parameters are the :math:`\beta` parameter, set
# by the sphericity of the Earth, and the Froude number, the relative
# importance of rotation to stratification. ::

beta = Constant('1.0')
F    = Constant('1.0')

# Additionally, we can create some Functions to store the eigenmodes. ::

eigenmodes_real, eigenmodes_imag = Function(Vcg), Function(Vcg)

# We define the Test Function :math:`\phi` and the Trial Function
# :math:`\psi` in our function space. ::

phi, psi = TestFunction(Vcg), TrialFunction(Vcg)

# To build the weak formulation of our equation we need to build two PETSc
# matrices in the form of a generalized eigenvalue problem,
# :math:`A\psi = \lambda M\psi`. We impose the boundary conditions on the
# mass matrix :math:`M`, since that is where we used integration by parts. ::

a =  beta*phi*psi.dx(0)*dx
m = -inner(grad(psi), grad(phi))*dx - F*psi*phi*dx
petsc_a = assemble(a).M.handle
petsc_m = assemble(m, bcs=bc).M.handle

# We can declare how many eigenpairs, eigenfunctions and eigenvalues, we
# want to find ::

num_eigenvalues = 1

# Next we will impose parameters onto our eigenvalue solver. The first is
# specifying that we have an generalized eigenvalue problem that is
# nonhermitian. The second specifies the spectral transform shift factor
# to be non-zero. The third requires we use a Krylov-Schur method,
# which is the default so this is not strictly necessary. Then, we ask for
# the eigenvalues with the largest imaginary part. Finally, we specify the
# tolerance. ::

opts = PETSc.Options()
opts.setValue("eps_gen_non_hermitian", None)
opts.setValue("st_pc_factor_shift_type", "NONZERO")
opts.setValue("eps_type", "krylovschur")
opts.setValue("eps_largest_imaginary", None)
opts.setValue("eps_tol", 1e-10)

# Finally, we build our eigenvalue solver using SLEPc. We add our PETSc
# matrices into the solver as operators and use setFromOptions() to call
# the PETSc parameters we previously declared. ::

es = SLEPc.EPS().create(comm=COMM_WORLD)
es.setFromOptions()
es.setDimensions(num_eigenvalues)
es.setOperators(petsc_a, petsc_m)

es.solve()

# Additionally we can find the number of converged eigenvalues. ::

nconv = es.getConverged()

# We now get the real and imaginary parts of the eigenvalue and
# eigenvector for the leading eigenpair (that with the largest in
# magnitude imaginary part).  First we check if we actually managed to
# converge any eigenvalues at all. ::

if nconv == 0:
    import sys
    warning("Did not converge any eigenvalues")
    sys.exit(0)

# If we did, we go ahead and extract them from the SLEPc eigenvalue
# solver::

vr, vi = petsc_a.getVecs()

lam = es.getEigenpair(0, vr, vi)

# and we gather the final eigenfunctions ::

eigenmodes_real.vector()[:], eigenmodes_imag.vector()[:] = vr, vi

# We can now list and show plots for the eigenvalues and eigenfunctions
# that were found. ::

print("Leading eigenvalue is:", lam)

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots()
    colors = tripcolor(eigenmodes_real, axes=axes)
    fig.colorbar(colors)

    fig, axes = plt.subplots()
    colors = tripcolor(eigenmodes_imag, axes=axes)
    fig.colorbar(colors)
except ImportError:
    warning("Matplotlib not available, not plotting eigemodes")
