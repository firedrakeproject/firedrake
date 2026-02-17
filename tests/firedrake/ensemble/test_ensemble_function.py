import numpy as np
import pytest
from pytest_mpi.parallel_assert import parallel_assert

import firedrake as fd


def random_func(f):
    f.dat.data_wo[...] = np.random.rand(*(f.dat.data.shape))
    return f


def random_efunc(f):
    for sub in f.subfunctions:
        random_func(sub)
    return f


def assign_scalar(u, s):
    for v in u.subfunctions:
        v.dat.data_wo[...] = s
    return u


@pytest.fixture
def ensemble():
    nspace = 2 if fd.COMM_WORLD.size == 4 else 1
    return fd.Ensemble(fd.COMM_WORLD, nspace)


@pytest.fixture
def mesh(ensemble):
    return fd.UnitSquareMesh(
        2, 2, comm=ensemble.comm,
        distribution_parameters={"partitioner_type": "simple"})


scalar_elements = {
    'CG': fd.FiniteElement('CG', cell=fd.triangle, degree=1),
    'BDM': fd.FiniteElement('BDM', cell=fd.triangle, degree=2),
    'V-RT': fd.VectorElement('RT', cell=fd.triangle, degree=1, dim=2),
    'T-DG': fd.TensorElement('DG', cell=fd.triangle, degree=1, shape=(2, 3))
}

# Test EnsembleFunction with 6 subfunctions with the
# elements below, distributed over 1 or more processors.
#
# This element sequence hits a variety of cases:
# 1. scalar, vector-valued, vector, tensor, mixed elements
# 2. mixed elements with non-scalar, components
# 3. identical adjacent components (4, 5)
# 4. mixed element with a single component (2)
# 5. mixed element with repeated component (6)
# 6. mixed element where the first component matches the previous element (5, 6)

elements = [
    scalar_elements['CG'],  # 1
    fd.MixedElement([scalar_elements[e] for e in ('CG', 'T-DG')]),  # 2
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


@pytest.fixture(params=["primal", "dual"])
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
def test_efunc_zero(ensemblefunc):
    """
    Test setting all components to zero.
    """
    # assign some nonzero value
    assign_scalar(ensemblefunc, 1)

    # check the norm is nonzero
    failed = []
    for i, u in enumerate(ensemblefunc.subfunctions):
        with u.dat.vec_ro as uvec:
            if uvec.norm() < 1e-14:
                failed.append(i)

    parallel_assert(
        len(failed) == 0,
        msg=("This test needs a nonzero initial value."
             f"The following subcomponents failed: {failed}")
    )

    ensemblefunc.zero()

    failed = []
    for i, u in enumerate(ensemblefunc.subfunctions):
        with u.dat.vec_ro as uvec:
            if uvec.norm() > 1e-14:
                failed.append(i)

    parallel_assert(
        len(failed) == 0,
        msg=("EnsembleFunction.zero should zero all components."
             f"The following subcomponents failed: {failed}")
    )


@pytest.mark.parallel(nprocs=[1, 2, 4, 6])
def test_efunc_zero_with_subset(ensemblefunc):
    """
    Test setting a subset of all components to zero.
    """

    # assign some nonzero value
    nonzero = 1
    assign_scalar(ensemblefunc, nonzero)

    # Functions on mixed function spaces don't accept the
    # subset argument, so we pass ... in those slots to
    # have the subset argument ignored for those subcomponents.
    subsets = [Ellipsis if type(V.ufl_element()) is fd.MixedElement else [0, 1]
               for V in ensemblefunc.function_space().local_spaces]

    ensemblefunc.zero(subsets)

    failed_zero_all = []
    failed_zero_subset = []
    failed_nonzero_notsubset = []
    for i, (u, subset) in enumerate(zip(ensemblefunc.subfunctions, subsets)):
        if subset is Ellipsis:
            with u.dat.vec_ro as uvec:
                if uvec.norm() > 1e-14:
                    failed_zero_all.append(i)
        else:
            if not np.allclose(u.dat.data_ro[:2], 0):
                failed_zero_subset.append(i)
            if not np.allclose(u.dat.data_ro[2:], nonzero):
                failed_nonzero_notsubset.append(i)

    parallel_assert(
        len(failed_zero_all) == 0,
        msg=("EnsembleFunction.zero() should zero the entire Function."
             f"The following subcomponents failed: {failed_zero_all}")
    )

    parallel_assert(
        len(failed_zero_subset) == 0,
        msg=("EnsembleFunction.zero(subset) should zero the subset."
             f"The following subcomponents failed: {failed_zero_subset}")
    )

    parallel_assert(
        len(failed_nonzero_notsubset) == 0,
        msg=("EnsembleFunction.zero(subset) should not zero outside the subset."
             f"The following subcomponents failed: {failed_nonzero_notsubset}")
    )


@pytest.mark.parallel(nprocs=[1, 2, 4, 6])
def test_efunc_subfunctions(ensemblefunc):
    """
    Test setting components of the local mixed space.
    """
    efunc = ensemblefunc
    espace = efunc.function_space()

    # Do the EnsembleFunction subfunctions have the right FunctionSpace?
    failed = []
    for i, (u, fs) in enumerate(zip(efunc.subfunctions,
                                    espace.local_spaces)):
        if u.function_space() != fs:
            failed.append(i)

    parallel_assert(
        len(failed) == 0,
        msg=("EnsembleFunction.subfunctions should have the same"
             " FunctionSpaces as EnsembleFunctionSpace.local_spaces."
             f"The following subfunctions failed: {failed}")
    )

    local_funcs = [random_func(fd.Function(V))
                   for V in espace.local_spaces]

    # Do the EnsembleFunction subfunctions view the right data?
    for usub, esub in zip([us for u in local_funcs for us in u.subfunctions],
                          efunc._full_local_function.subfunctions):
        esub.assign(usub)

    for i, (u, e) in enumerate(zip(local_funcs, efunc.subfunctions)):
        with u.dat.vec_ro as uvec, e.dat.vec_ro as evec:
            if not np.allclose(uvec.array_r, evec.array_r):
                failed.append(i)

    parallel_assert(
        len(failed) == 0,
        msg=("EnsembleFunction.subfunctions should view the data in"
             " EnsembleFunction._full_local_function."
             f"The following subfunctions failed: {failed}")
    )


@pytest.mark.parallel(nprocs=[1, 2, 4, 6])
def test_efunc_riesz_representation(ensemblefunc):
    """
    Test that taking the riesz representation of an EnsembleFunction
    is the same as taking the riesz representation of each component.
    """
    efunc = random_efunc(ensemblefunc)
    edual = ensemblefunc.riesz_representation()

    parallel_assert(
        edual.function_space() == efunc.function_space().dual(),
        msg=("The EnsembleFunctionSpace of EnsembleFunction.dual()"
             " should be EnsembleFunction.function_space().dual()")
    )

    failed = []
    for i, (ef, ed) in enumerate(zip(efunc.subfunctions, edual.subfunctions)):
        check = ef.riesz_representation()
        with ed.dat.vec_ro as edvec, check.dat.vec_ro as cvec:
            if not np.allclose(edvec.array_r, cvec.array_r):
                failed.append(i)

    parallel_assert(
        len(failed) == 0,
        msg=("EnsembleFunction.riesz_representation() give the same"
             " values as the riesz_representation() of each subfunction."
             f" The following subfunctions failed: {failed}")
    )


@pytest.mark.parallel(nprocs=[1, 2, 4, 6])
def test_efunc_assign(ensemblefunc):
    """
    Test assigning one EnsembleFunction to another.
    """
    efunc0 = random_efunc(ensemblefunc)
    efunc1 = efunc0.copy().zero()

    efunc1.assign(efunc0)

    # Do the EnsembleFunction subfunctions match?
    failed = []
    for i, (u0, u1) in enumerate(zip(efunc0.subfunctions,
                                     efunc1.subfunctions)):
        with u0.dat.vec_ro as v0, u1.dat.vec_ro as v1:
            if not np.allclose(v0.array_r, v1.array_r):
                failed.append(i)

    parallel_assert(
        len(failed) == 0,
        msg=("EnsembleFunction.assign should copy all subfunctions."
             f"The following subfunctions failed: {failed}")
    )


@pytest.mark.parallel(nprocs=[1, 2, 4, 6])
def test_efunc_copy(ensemblefunc):
    """
    Test copying one EnsembleFunction to another.
    """
    efunc0 = random_efunc(ensemblefunc)
    efunc1 = efunc0.copy()

    # Do the EnsembleFunction subfunctions match?
    failed = []
    for i, (u0, u1) in enumerate(zip(efunc0.subfunctions,
                                     efunc1.subfunctions)):
        with u0.dat.vec_ro as v0, u1.dat.vec_ro as v1:
            if not np.allclose(v0.array_r, v1.array_r):
                failed.append(i)

    parallel_assert(
        len(failed) == 0,
        msg=("EnsembleFunction.copy should copy all subfunctions."
             f"The following subfunctions failed: {failed}")
    )
