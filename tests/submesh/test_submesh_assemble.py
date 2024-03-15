import pytest
import numpy as np
from firedrake import *
from firedrake.cython import dmcommon


def test_submesh_assemble_mixed_scalar():
    dim = 2
    mesh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(conditional(x > 1., 1, 0))
    mesh.mark_entities(indicator_function, 999)
    subm = Submesh(mesh, dmcommon.CELL_SETS_LABEL, 999, mesh.topological_dimension())
    subm.init()
    V0 = FunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(subm, "CG", 1)
    V = V0 * V1
    u = TrialFunction(V)
    v = TestFunction(V)
    u0, u1 = split(u)
    v0, v1 = split(v)
    dx0 = Measure("cell", domain=mesh)
    dx1 = Measure("cell", domain=subm)
    a = inner(u1, v0) * dx0(999) + inner(u0, v1) * dx1
    A = assemble(a, mat_type="nest")
    assert np.allclose(A.M.sparsity[0][0].nnz, [1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(A.M.sparsity[0][1].nnz, [4, 4, 4, 4, 0, 0])
    assert np.allclose(A.M.sparsity[1][0].nnz, [4, 4, 4, 4])
    assert np.allclose(A.M.sparsity[1][1].nnz, [1, 1, 1, 1])  # bc nodes
    M10 = np.array([[1./9. , 1./18., 1./36., 1./18., 0., 0.],
                    [1./18., 1./9. , 1./18., 1./36., 0., 0.],
                    [1./36., 1./18., 1./9. , 1./18., 0., 0.],
                    [1./18., 1./36., 1./18., 1./9. , 0., 0.]])
    assert np.allclose(A.M[0][1].values, np.transpose(M10))
    assert np.allclose(A.M[1][0].values, M10)
