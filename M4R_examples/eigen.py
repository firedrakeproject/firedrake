# Define the variational form
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx

# Apply the homogeneous Dirichlet boundary conditions
bc = DirichletBC(V, 0.0, "on_boundary")

# Create eigenproblem
eigenprob = LinearEigenproblem(a, bcs=bc)

# Create corresponding eigensolver, looking for 1 eigenvalue
eigensolver = LinearEigensolver(eigenprob, 1)

# Solve the problem
eigensolver.solve()

# View eigenvalues and eigenvectors
evals = eigensolver.eigenvalues()
evecs = eigensolver.eigenvectors()
