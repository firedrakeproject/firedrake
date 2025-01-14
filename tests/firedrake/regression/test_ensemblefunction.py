import firedrake as fd
import pytest
import numpy as np


def random_func(f):
    for dat in f.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))


def assign_scalar(u, s):
    for v in u.subfunctions:
        for dat in v.dat:
            dat.data[:] = s
    return u


def norm(u):
    return sum(np.sum(np.abs(usubdat.data))
               for usub in u.subfunctions
               for usubdat in usub.dat)


@pytest.fixture
def ensemble():
    if fd.COMM_WORLD.size == 1:
        return fd.Ensemble(fd.COMM_WORLD, 1)
    return fd.Ensemble(fd.COMM_WORLD, 2)


@pytest.fixture
def mesh(ensemble):
    return fd.UnitSquareMesh(2, 2, comm=ensemble.comm)


elements = [
    fd.FiniteElement('CG', cell=fd.triangle, degree=1),
    fd.FiniteElement('BDM', cell=fd.triangle, degree=2),
    fd.FiniteElement('DG', cell=fd.triangle, degree=0),
    fd.VectorElement('RT', cell=fd.triangle, degree=1, dim=2),
    fd.FiniteElement('CG', cell=fd.triangle, degree=1),
    fd.TensorElement('DG', cell=fd.triangle, degree=1, shape=(2, 3))
]


@pytest.fixture
def Wlocal(ensemble, mesh):
    ensemble_size = ensemble.ensemble_comm.size
    ensemble_rank = ensemble.ensemble_comm.rank
    nelems = len(elements)
    assert nelems % ensemble_size == 0
    nlocals = nelems // ensemble_size
    offset = ensemble_rank*nlocals
    return [fd.FunctionSpace(mesh, elements[i])
            for i in range(offset, offset+nlocals)]


function_type = ["EnsembleFunction", "EnsembleCofunction"]


@pytest.fixture(params=function_type)
def ensemblefunc(request, ensemble, Wlocal):
    if request.param == 'EnsembleFunction':
        return fd.EnsembleFunction(ensemble, Wlocal)
    elif request.param == 'EnsembleCofunction':
        Wduals = [W.dual() for W in Wlocal]
        return fd.EnsembleCofunction(ensemble, Wduals)
    else:
        raise ValueError(f"Unknown ensemblefunc type {request.param}")


@pytest.mark.parallel(nprocs=[1, 2, 6])
def test_zero(ensemblefunc):
    """
    Test setting all components to zero.
    """
    # assign some nonzero value
    assign_scalar(ensemblefunc, 1)

    # check the norm is nonzero
    for u in ensemblefunc.subfunctions:
        assert norm(u) > 1e-14, "This test needs a nonzero initial value."

    ensemblefunc.zero()

    for u in ensemblefunc.subfunctions:
        assert norm(u) < 1e-14, "EnsembleFunction.zero should zero all components"


@pytest.mark.parallel(nprocs=[1, 2, 6])
def test_zero_with_subset(ensemblefunc):
    """
    Test setting a subset of all components to zero.
    """
    from pyop2 import Subset

    # assign some nonzero value
    nonzero = 1
    assign_scalar(ensemblefunc, nonzero)

    subsets = [Subset(functionspace.node_set, [0, 1])
               for functionspace in ensemblefunc.local_function_spaces]

    ensemblefunc.zero(subsets)

    for u, subset in zip(ensemblefunc.subfunctions, subsets):
        assert np.allclose(u.dat.data_ro[:2], 0), "EnsembleFunction.zero(subset) should zero the subset"
        assert np.allclose(u.dat.data_ro[2:], nonzero), "EnsembleFunction.zero(subset) should only modify the subset"
