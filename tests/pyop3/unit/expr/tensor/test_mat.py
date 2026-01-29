import numpy as np
import pytest

import pyop3 as op3


@pytest.fixture
def petsc_mat():
    axes = op3.AxisTree.from_iterable([3, 2])
    sparsity = op3.Mat.sparsity(axes, axes)
    # dense
    map_ = np.arange(6, dtype=op3.IntType)
    values = np.full((6, 6), 666, dtype=op3.ScalarType)
    sparsity.petscmat.setValues(map_, map_, values)
    return op3.Mat.from_sparsity(sparsity)


@pytest.fixture
def array_mat():
    axes = op3.AxisTree.from_iterable([3, 2])
    buffer = op3.ArrayBuffer(np.zeros(6*6))
    return op3.Mat.empty(axes, axes, buffer=buffer)


@pytest.mark.parametrize("mat", [petsc_mat, array_mat])
def test_zero(mat):
    raise NotImplementedError
    mat.zero()
    assert np.allclose(mat.values, 0)


@pytest.mark.parametrize("mat", [petsc_mat, array_mat])
def test_eager_assign(mat):
    expr = mat.assign(1, eager=True)
    assert np.allclose(mat.values, 1)
    assert expr is None


@pytest.mark.parametrize("mat", [petsc_mat, array_mat])
def test_lazy_assign(mat):
    expr = mat.assign(1)
    assert np.allclose(mat.values, 0)

    expr()
    assert np.allclose(mat.values, 1)


@pytest.mark.parametrize("mat", [petsc_mat, array_mat])
def test_subset_assign(mat):
    mat[(slice(step=2), 1), (slice(step=2), 1)].assign()
    raise NotImplementedError
    assert np.allclose(mat.values, 0)
