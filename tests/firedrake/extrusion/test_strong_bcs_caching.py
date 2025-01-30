import numpy as np
from firedrake import *


def test_extrusion_strong_bcs_caching(extmesh):
    m = extmesh(1, 1, 1)

    V = FunctionSpace(m, "CG", 1)

    bc1 = DirichletBC(V, 0, "bottom")
    bc2 = DirichletBC(V, 1, "top")

    v = TestFunction(V)
    u = TrialFunction(V)

    a = inner(u, v)*dx

    Aboth = assemble(a, bcs=[bc1, bc2])
    Aneither = assemble(a)
    Abottom = assemble(a, bcs=[bc1])
    Atop = assemble(a, bcs=[bc2])

    # None of the matrices should be the same
    assert not np.allclose(Aboth.M.values, Aneither.M.values)
    assert not np.allclose(Aboth.M.values, Atop.M.values)
    assert not np.allclose(Aboth.M.values, Abottom.M.values)
    assert not np.allclose(Aneither.M.values, Atop.M.values)
    assert not np.allclose(Aneither.M.values, Abottom.M.values)
    assert not np.allclose(Atop.M.values, Abottom.M.values)

    # There should be no zeros on the diagonal
    assert not any(Atop.M.values.diagonal() == 0)
    assert not any(Abottom.M.values.diagonal() == 0)
    assert not any(Aneither.M.values.diagonal() == 0)
    # The top/bottom case should just be the identity (since the only
    # dofs live on the top and bottom)
    assert np.allclose(Aboth.M.values, np.diag(np.ones_like(Aboth.M.values.diagonal())))
