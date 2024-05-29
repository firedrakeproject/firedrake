import pytest
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc


def topetsc(A):
    return A.petscmat


def test_laplace_physical_ev(parallel=False):
    try:
        from slepc4py import SLEPc
    except ImportError:
        pytest.skip(reason="SLEPc unavailable, skipping eigenvalue test")

    mesh = UnitSquareMesh(64, 64)
    V = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, Constant(0.0), (1, 2, 3, 4))

    # We just need the Stiffness and Mass matrix
    a = inner(grad(u), grad(v))*dx
    m = inner(u, v)*dx

    A = topetsc(assemble(a, bcs=[bc], weight=1.))
    M = topetsc(assemble(m, bcs=[bc], weight=0.))

    # Another way to shift the eigenvalues of value 1.0 out of
    # the spectrum of interest:
    # vals = np.repeat(1E8, len(bc.nodes))
    # A.setValuesLocalRCV(bc.nodes.reshape(-1, 1),
    #                     bc.nodes.reshape(-1, 1),
    #                     vals.reshape(-1, 1))

    E = SLEPc.EPS()
    E.create(comm=mesh.comm)
    E.setOperators(A, M)
    st = E.getST()
    st.setType('sinvert')
    kspE = st.getKSP()
    kspE.setType('fgmres')
    E.setDimensions(5, PETSc.DECIDE)
    E.solve()

    nconv = E.getConverged()
    assert nconv > 0

    # Create the results vectors
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()
    ev = []
    for i in range(nconv):
        k = E.getEigenpair(i, vr, vi)
        ev.append(k.real)

    # Exact eigenvalues are
    ev_exact = np.array([1**2 * np.pi**2 + 1**2 * np.pi**2,
                         2**2 * np.pi**2 + 1**2 * np.pi**2,
                         1**2 * np.pi**2 + 2**2 * np.pi**2])
    assert np.allclose(ev_exact, np.array(ev)[:3], atol=1e-1)


@pytest.mark.parallel
def test_laplace_parallel():
    test_laplace_physical_ev(parallel=True)
