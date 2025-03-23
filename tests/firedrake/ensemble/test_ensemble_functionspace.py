import firedrake as fd
import pytest
from pytest_mpi.parallel_assert import parallel_assert


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


space_types = ("primal", "dual")


@pytest.fixture(params=space_types)
def ensemblespace(request, ensemble, Wlocal):
    if request.param == 'primal':
        return fd.EnsembleFunctionSpace(Wlocal, ensemble)
    elif request.param == 'dual':
        return fd.EnsembleDualSpace([V.dual() for V in Wlocal], ensemble)
    else:
        raise ValueError(f"Unknown function space type {request.param}")


@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.parametrize("space_type", space_types)
def test_ensemble_space_reject_inconsistent_meshes(ensemble, space_type):
    """
    EnsembleFunctionSpace should reject lists of local FunctionSpaces
    that are defined on different meshes (on any ensemble rank).
    """
    mesh0 = fd.UnitIntervalMesh(
        4, comm=ensemble.comm,
        distribution_parameters={"partitioner_type": "simple"})

    if ensemble.ensemble_rank == (ensemble.ensemble_size - 1):
        mesh1 = fd.UnitIntervalMesh(
            4, comm=ensemble.comm,
            distribution_parameters={"partitioner_type": "simple"})
    else:
        mesh1 = mesh0

    FS0 = fd.FunctionSpace(mesh0, 'CG', 1)
    FS1 = fd.FunctionSpace(mesh1, 'CG', 1)

    if space_type == 'primal':
        spaces = (FS0, FS1)
    elif space_type == 'dual':
        spaces = (FS0.dual(), FS1.dual())
    else:
        raise ValueError(f"Unrecognised {space_type=}")

    with pytest.raises(ValueError):
        _ = fd.EnsembleFunctionSpace(spaces, ensemble)


@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.parametrize("space_type", space_types)
def test_ensemble_space_reject_inconsistent_spacetype(ensemble, mesh, space_type):
    """
    EnsembleFunctionSpace should reject lists of local FunctionSpaces
    that are not all primal or all dual (on any ensemble rank).
    """
    CG1 = fd.FunctionSpace(mesh, 'CG', 1)
    DG0 = fd.FunctionSpace(mesh, 'DG', 0)

    if ensemble.ensemble_rank == (ensemble.ensemble_size - 1):
        spaces = (CG1, DG0.dual())
    else:
        if space_type == 'primal':
            spaces = (CG1, DG0)
        else:
            spaces = (CG1.dual(), DG0.dual())

    with pytest.raises(ValueError):
        if space_type == 'primal':
            _ = fd.EnsembleFunctionSpace(spaces, ensemble)
        elif space_type == 'dual':
            _ = fd.EnsembleDualSpace(spaces, ensemble)


@pytest.mark.parallel(nprocs=[1, 2])
def test_ensemble_dualspace_reject_primal_spaces(ensemble, mesh):
    """
    EnsembleDualSpace should reject lists of primal local FunctionSpaces.
    """
    CG1 = fd.FunctionSpace(mesh, 'CG', 1)
    DG0 = fd.FunctionSpace(mesh, 'DG', 0)

    if ensemble.ensemble_rank == (ensemble.ensemble_size - 1):
        spaces = (CG1, DG0)
    else:
        spaces = (CG1.dual(), DG0.dual())

    with pytest.raises(ValueError):
        _ = fd.EnsembleDualSpace(spaces, ensemble)


def test_ensemble_space_allows_dual_init(ensemble, mesh):
    """
    EnsembleFunctionSpace should return an EnsembleDualSpace if
    given a list of dual local FunctionSpaces.
    """
    CG1 = fd.FunctionSpace(mesh, 'CG', 1)
    DG0 = fd.FunctionSpace(mesh, 'DG', 0)

    spaces = (CG1.dual(), DG0.dual())

    efs = fd.EnsembleFunctionSpace(spaces, ensemble)

    parallel_assert(lambda: isinstance(efs, fd.EnsembleDualSpace))


@pytest.mark.parallel(nprocs=[1, 2, 4, 6])
def test_ensemble_local_spaces_correct(ensemblespace, Wlocal):
    """
    The local_spaces of EnsembleFunctionSpace should
    match the ones provided at instantiation.
    """
    efs = ensemblespace
    ensemble = efs.ensemble

    # compare to the right type of space
    if isinstance(efs, fd.EnsembleDualSpace):
        Wlocal = [W.dual() for W in Wlocal]

    # Does efs have the correct number of spaces locally?
    parallel_assert(lambda: len(efs.local_spaces) == len(Wlocal))
    parallel_assert(lambda: efs.nlocal_spaces == len(efs.local_spaces))

    # Correct number globally?
    nglobal = ensemble.ensemble_comm.allreduce(efs.nlocal_spaces)
    parallel_assert(lambda: efs.nglobal_spaces == nglobal)

    local_space_matches = [
        V == W for V, W in zip(efs.local_spaces, Wlocal)]

    # each rank could have a different number of spaces, so check
    # rank-by-rank not space-by-space
    for root in range(ensemble.ensemble_size):
        parallel_assert(
            lambda: all(local_space_matches),
            participating=(ensemble.ensemble_rank == root),
            msg=f"{ensemble.ensemble_rank=}, {local_space_matches=}")


@pytest.mark.parallel(nprocs=[1, 2, 4, 6])
def test_ensemble_dofsizes_correct(ensemblespace):
    """
    The number of dofs of an EnsembleFunctionSpace
    is the sum of the dofs of the local_spaces.
    """
    efs = ensemblespace
    ensemble = efs.ensemble
    rank = ensemble.global_comm.rank

    nlocal_rank_dofs = sum(fs.dof_dset.layout_vec.getLocalSize()
                           for fs in efs.local_spaces)
    nlocal_comm_dofs = ensemble.comm.allreduce(nlocal_rank_dofs)
    nglobal_dofs = ensemble.ensemble_comm.allreduce(nlocal_comm_dofs)

    parallel_assert(
        lambda: efs.nlocal_rank_dofs == nlocal_rank_dofs,
        msg=f"{rank=}, {efs.nlocal_rank_dofs=}, {nlocal_rank_dofs=}")

    parallel_assert(
        lambda: efs.nlocal_comm_dofs == nlocal_comm_dofs,
        msg=f"{rank=}, {efs.nlocal_comm_dofs=}, {nlocal_comm_dofs=}")

    parallel_assert(
        lambda: efs.nglobal_dofs == nglobal_dofs,
        msg=f"{rank=}, {efs.nglobal_dofs=}, {nglobal_dofs=}")


@pytest.mark.parallel(nprocs=[1, 2, 4, 6])
def test_ensemble_space_dual(ensemblespace):
    """
    EnsembleFunctionSpace.dual() should return the
    dual type propogated to the subcomponents, and
    efs.dual().dual() should return the original.
    """
    ensemble = ensemblespace.ensemble

    if type(ensemblespace) is fd.EnsembleFunctionSpace:
        dual_type = fd.EnsembleDualSpace
    else:
        dual_type = fd.EnsembleFunctionSpace

    dual = ensemblespace.dual()
    parallel_assert(lambda: type(dual) is dual_type)

    # are the dual subspaces correct?
    local_space_matches = [
        V == W.dual() for V, W in zip(dual.local_spaces, ensemblespace.local_spaces)
    ]
    # each rank could have a different number of spaces, so check
    # rank-by-rank not space-by-space
    ensemble_rank = ensemble.ensemble_rank
    for root in range(ensemble.ensemble_size):
        parallel_assert(
            lambda: all(local_space_matches),
            participating=(ensemble.ensemble_rank == root),
            msg=f"{ensemble_rank=}, {local_space_matches=}")

    dual2 = dual.dual()
    parallel_assert(lambda: type(dual.dual()) is type(ensemblespace))

    # are the dual subspaces the originals?
    local_space_matches = [
        V == W for V, W in zip(dual2.local_spaces, ensemblespace.local_spaces)
    ]
    # each rank could have a different number of spaces, so check
    # rank-by-rank not space-by-space
    ensemble_rank = ensemblespace.ensemble.ensemble_rank
    for root in range(ensemble.ensemble_size):
        parallel_assert(
            lambda: all(local_space_matches),
            participating=(ensemble.ensemble_rank == root),
            msg=f"{ensemble_rank=}, {local_space_matches=}")


@pytest.mark.parallel(nprocs=[1, 2, 4])
def test_ensemble_space_equality(ensemblespace):
    """
    EnsembleFunctionSpaces should only compare equal if of
    the same type (primal vs dual) and all subspaces are equal.
    """
    ensemble = ensemblespace.ensemble
    orig_spaces = ensemblespace.local_spaces

    # .dual().dual() should be a round trip
    parallel_assert(lambda: ensemblespace == ensemblespace)
    parallel_assert(lambda: ensemblespace != ensemblespace.dual())
    parallel_assert(lambda: ensemblespace == ensemblespace.dual().dual())

    # Duplicate is equal
    dup_fs = fd.EnsembleFunctionSpace(orig_spaces, ensemble)
    parallel_assert(lambda: ensemblespace == dup_fs)

    # Same length, different space is not equal
    if ensemble.ensemble_rank == (ensemble.ensemble_size - 1):
        diff_spaces = [orig_spaces[1], *orig_spaces[1:]]
    else:
        diff_spaces = orig_spaces

    diff_fs = fd.EnsembleFunctionSpace(diff_spaces, ensemble)
    parallel_assert(lambda: ensemble != diff_fs)

    # Same spaces, shorter length is not equal
    if ensemble.ensemble_rank == 0:
        short_spaces = orig_spaces[1:]
    else:
        short_spaces = orig_spaces

    short_fs = fd.EnsembleFunctionSpace(short_spaces, ensemble)
    parallel_assert(lambda: ensemble != short_fs)

    # Same spaces, longer length is not equal
    if ensemble.ensemble_rank == (ensemble.ensemble_size - 1):
        long_spaces = [*orig_spaces, orig_spaces[-1]]
    else:
        long_spaces = orig_spaces

    long_fs = fd.EnsembleFunctionSpace(long_spaces, ensemble)
    parallel_assert(lambda: ensemble != long_fs)
