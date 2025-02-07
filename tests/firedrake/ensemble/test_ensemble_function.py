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


scalar_elements = {
    'CG': fd.FiniteElement('CG', cell=fd.triangle, degree=1),
    'BDM': fd.FiniteElement('BDM', cell=fd.triangle, degree=2),
    'V-RT': fd.VectorElement('RT', cell=fd.triangle, degree=1, dim=2),
    'T-DG': fd.TensorElement('DG', cell=fd.triangle, degree=1, shape=(2, 3))
}

# Test an EnsembleFunction with 8 subfunctions with the elements below, distributed over 1 or more processors.
# This element sequence below hits a variety of cases:
# - scalar, vector-valued, vector, tensor, mixed elements
# - mixed elements with scalar, vector-valued, vector and tensor components
# - repeated adjacent components (6, 7)
# - repeated non-adjacent components (2, 6)
# - mixed element with a single component (8)
# - mixed element with repeated component (5)
# - mixed element where the first component matches the previous element (2, 3)

elements = [
    scalar_elements['CG'],  # 1
    scalar_elements['BDM'],  # 2
    fd.MixedElement([scalar_elements[e] for e in ('BDM', 'CG')]),  # 3
    scalar_elements['T-DG'],  # 4
    fd.MixedElement([scalar_elements[e] for e in ('V-RT', 'CG', 'CG')]),  # 5
    scalar_elements['BDM'],  # 6
    scalar_elements['BDM'],  # 7
    fd.MixedElement([scalar_elements[e] for e in ('T-DG',)])  # 8
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


space_type = ["primal", "dual"]


@pytest.fixture(params=space_type)
def ensemblespace(request, ensemble, Wlocal):
    if request.param == 'primal':
        return fd.EnsembleFunctionSpace(Wlocal, ensemble)
    elif request.param == 'dual':
        return fd.EnsembleDualSpace([V.dual() for V in Wlocal], ensemble)
    else:
        raise ValueError(f"Unknown function space type {request.param}")


@pytest.fixture
def ensemblefunc(ensemblespace):
    return fd.EnsembleFunction(ensemblespace)


@pytest.mark.parallel(nprocs=[1, 2, 4, 8])
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


@pytest.mark.parallel(nprocs=[1, 2, 4, 8])
def test_zero_with_subset(ensemblefunc):
    """
    Test setting a subset of all components to zero.
    """
    from pyop2 import Subset

    # assign some nonzero value
    nonzero = 1
    assign_scalar(ensemblefunc, nonzero)

    subsets = [None if type(V.ufl_element()) is fd.MixedElement else Subset(V.node_set, [0, 1])
               for V in ensemblefunc.function_space().local_spaces]

    ensemblefunc.zero(subsets)

    for u, subset in zip(ensemblefunc.subfunctions, subsets):
        if subset is None:
            continue
        assert np.allclose(u.dat.data_ro[:2], 0), "EnsembleFunction.zero(subset) should zero the subset"
        assert np.allclose(u.dat.data_ro[2:], nonzero), "EnsembleFunction.zero(subset) should only modify the subset"
