# Define the variational form
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx

# Apply the boundary conditions
bc = None # to be added

# Specify how many eigenvalues we search for
n_evals = 1

# Create the Linear Eigenproblem
eigenprob = LinearEigenproblem(a, bcs=bc)

# Create the Linear Eigensolver
eigensolver = LinearEigensolver(eigenprob, n_evals)

# Solve the problem
eigensolver.solve()

# View eigenvalues and eigenvectors
evals = eigensolver.eigenvalues()
evecs = eigensolver.eigenvectors()
