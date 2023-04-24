# Define the variational form
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx

# Define the linear form for the RHS of the equation
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate(sin(x)*cos(y))
L = inner(f, v) * dx 

# Apply the boundary conditions
bc = None # to be added

 # Create a function to hold the solution
u_soln = Function(V) 

# Create the Linear Variational Problem
problem = LinearVariationalProblem(a, L, u_soln, bcs=bcs)

# Create the Linear Variational Solver
solver = LinearVariationalSolver(problem)  

# Solve the problem
solver.solve()
