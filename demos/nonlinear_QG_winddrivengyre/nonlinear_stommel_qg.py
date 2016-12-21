from firedrake import *
import numpy as np
import ufl, os

# GEOMETRY
n0   = 50                                            # Spatial resolution
Ly   = 1.0                                           # Meridonal length
Lx   = np.sqrt(2)									 # Zonal length
mesh = RectangleMesh(n0, n0, Lx, Ly, reorder = None) 

# FUNCTION SPACES
Vcg  = FunctionSpace(mesh, "CG", 3)                    # CG elements for Streamfunction

# BOUNDARY CONDITIONS
bc = DirichletBC(Vcg, 0.0, "on_boundary")

# PHYSICAL PARAMETERS AND WINDS
beta = Constant("1.0")                                 # Beta parameter
F    = Constant("1.0")                                 # Burger number
r    = Constant("0.2")                                 # Bottom drag
tau  = Constant("0.001")                               # Wind Forcing
Qwinds = Function(Vcg).interpolate(Expression("-tau*cos(pi*( (x[1]/Ly)-0.5 ))", tau=tau, Ly=Ly))

# TEST AND TRIAL FUNCTIONS
phi, psi = TestFunction(Vcg), TrialFunction(Vcg)

psi_lin = Function(Vcg, name="Linear Streamfunction")
psi_non = Function(Vcg, name="Nonlinear Streamfunction")

#OPERATORS
gradperp = lambda i: as_vector((-i.dx(1),i.dx(0)))

# DEFINE WEAK FORM OF LINEAR PROBLEM
a = - r*inner(grad(psi), grad(phi))*dx - F*psi*phi*dx + beta*psi.dx(0)*phi*dx
L =  Qwinds*phi*dx

# SET-UP ELLIPTIC INVERTER FOR LINEAR PROBLEM
linear_problem = LinearVariationalProblem(a, L, psi_lin, bcs=bc)
linear_solver = LinearVariationalSolver(linear_problem,
		solver_parameters={
			'ksp_type':'preonly',
			'pc_type':'lu'})
linear_solver.solve()

p = plot(psi_lin)
p.show()

# USE LINEAR SOLUTION AS A GOOD GUESS
psi_non.assign(psi_lin)

# DEFINE WEAK FORM OF NONLINEAR PROBLEM
G = - inner(grad(phi),gradperp(psi_non))*div(grad(psi_non))*dx \
	- r*inner(grad(psi_non), grad(phi))*dx - F*psi_non*phi*dx \
	+ beta*psi_non.dx(0)*phi*dx \
	- Qwinds*phi*dx

# SET-UP ELLIPTIC INVERTER FOR NONLINEAR PROBLEM
nonlinear_problem = NonlinearVariationalProblem(G, psi_non, bcs=bc)
nonlinear_solver = NonlinearVariationalSolver(nonlinear_problem,
		solver_parameters={
			'snes_type': 'newtonls',
			'ksp_type':'preonly',
			'pc_type':'lu'})
nonlinear_solver.solve()

p = plot(psi_non)
p.show()

file = File('Nonlinear_Streamfunction.pvd')
file.write(psi_non)

tf, difference = TestFunction(Vcg), TrialFunction(Vcg)

a = difference*tf*dx
L = (psi_lin - psi_non)*tf*dx
difference = Function(Vcg, name="Difference")
solve(a==L, difference, None)

p = plot(difference)
p.show()

file = File("Difference between Linear and Nonlinear Streamfunction.pvd")
file.write(difference)
