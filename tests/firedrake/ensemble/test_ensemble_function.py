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
    nspace = 2 if fd.COMM_WORLD.size == 4 else 1
    return fd.Ensemble(fd.COMM_WORLD, nspace)


@pytest.fixture
def mesh(ensemble):
    return fd.UnitSquareMesh(2, 2, comm=ensemble.comm)


scalar_elements = {
    'CG': fd.FiniteElement('CG', cell=fd.triangle, degree=1),
    'BDM': fd.FiniteElement('BDM', cell=fd.triangle, degree=2),
    'V-RT': fd.VectorElement('RT', cell=fd.triangle, degree=1, dim=2),
    'T-DG': fd.TensorElement('DG', cell=fd.triangle, degree=1, shape=(2, 3))
}

# Test EnsembleFunction with 6 subfunctions with the elements below, distributed over 1 or more processors.
# This element sequence hits a variety of cases:
# - scalar, vector-valued, vector, tensor, mixed elements
# - mixed elements with non-scalar, components
# - identical adjacent components (4, 5)
# - mixed element with a single component (2)
# - mixed element with repeated component (6)
# - mixed element where the first component matches the previous element (5, 6)

elements = [
    scalar_elements['CG'],  # 1
    fd.MixedElement([scalar_elements[e] for e in ('T-DG',)]),  # 2
    scalar_elements['T-DG'],  # 3
    scalar_elements['BDM'],  # 4
    scalar_elements['BDM'],  # 5
    fd.MixedElement([scalar_elements[e] for e in ('BDM', 'V-RT', 'V-RT')])  # 6
]

# Test with four time partitions:
# time partitions - serial
# time partitions - parallel with P<N
# time partitions - parallel with P=N
# Also test with spatial parallelism to check that we don't mix up comms accidentally somewhere.
#
#       time      space      partition
# P1 : serial   | serial   | 6
# P2 : parallel | serial   | 2, 4
# P4 : parallel | parallel | 3, 3
# P6 : parallel | serial   | 1 per rank


@pytest.fixture
def Wlocal(ensemble, mesh):
    gcomm = ensemble.global_comm
    erank = ensemble.ensemble_rank
    esize = ensemble.ensemble_size
    if gcomm.size == 1:
        elems = elements
    elif gcomm.size == 2:
        if erank == 0:
            elems = elements[:2]
        if erank == 1:
            elems = elements[2:]
    elif gcomm.size == 4:
        nelem = len(elements)//esize
        offset = erank*nelem
        elems = elements[offset:offset+nelem]
    elif gcomm.size == 6:
        elems = [elements[erank]]
    else:
        raise ValueError("Invalid number of ranks")

    return [fd.FunctionSpace(mesh, e) for e in elems]


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


@pytest.mark.parallel(nprocs=[1, 2, 4, 6])
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


@pytest.mark.parallel(nprocs=[1, 2, 4, 6])
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
