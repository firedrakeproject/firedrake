
"""This demo program solves the Poisson eigenvalue problem

  - div grad u(x) = lambda u(x) (STRONG FORM)
  grad u(x) dot grad v(x) dx = lambda u(x)v(x) dx (WEAK FORM)

on the unit interval (0, pi)
"""
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

def main(n, k_range):
    # eigensolver
    mesh = IntervalMesh(n, 0, pi)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(grad(u), grad(v))) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    eigenprob = LinearEigenproblem(a, bcs=bc) #bcs=bc)
    eigensolver = LinearEigensolver(eigenprob, n)
    ncov = eigensolver.solve()

    # boffi
    h = pi / n
    x = np.linspace(0, np.pi, n+1)[1:-1] # exclude ends
    boffi_evals = (6/h**2) * (1-np.cos(x)) / (2+np.cos(x))


    # plotting
    fig, axs = plt.subplots(k_range, 1, figsize=(12, 8))
    for k in range(k_range):        
        # my results
        eval=eigensolver.eigenvalue(k)
        eigenmodes_real, _ = eigensolver.eigenfunction(k)
        evec = eigenmodes_real.vector()[1:-1] # scale by max norm
        max_evec_val = max(evec)
        evec_scaled = evec / max_evec_val

        # boffi
        boffi_evec = np.sin(k * x)
        boffi_eval = boffi_evals[k]
        
        # plot
        axs[k].plot(x, boffi_evec, label=f'Boffi eval = {boffi_eval:.2f}')
        axs[k].plot(x, evec_scaled, label=f'Firedrake eval = {(1/eval):.2f}')
        # axs[k].xlabel('x')
        # axs[k].ylabel('u')
        axs[k].legend()
        axs[k].set_title(f'Eigenspace for k = {k}')
    fig.tight_layout()
    plt.show()


n = 200
k = 4
main(n, k)



