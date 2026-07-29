import pytest
import numpy as np
import petsctools
from pytest_mpi.parallel_assert import parallel_assert
from firedrake import *
from firedrake.ensemble.ensemble_mat import EnsembleBlockDiagonalMat


@pytest.mark.parallel([1, 2, 3, 4])
def test_ensemble_mat():
    # create ensemble
    global_ranks = COMM_WORLD.size
    nspatial_ranks = 2 if (global_ranks % 2 == 0) else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)
    ensemble_rank = ensemble.ensemble_rank

    # create mesh
    mesh = UnitIntervalMesh(10, comm=ensemble.comm)

    # create function spaces
    CG = FunctionSpace(mesh, "CG", 1)
    DG = FunctionSpace(mesh, "DG", 1+ensemble_rank)

    # create ensemble function spaces / functions
    row_space = EnsembleFunctionSpace([CG, CG], ensemble)
    col_space = EnsembleFunctionSpace([CG, DG], ensemble)

    # build forms
    u, v = TrialFunction(CG), TestFunction(CG)
    nu = Constant(ensemble_rank+1)
    a0 = inner(u, v)*dx + nu*inner(grad(u), grad(v))*dx

    u, v = TrialFunction(CG), TestFunction(DG)
    a1 = (1/nu)*inner(u, v)*dx

    # assemble mats
    A0mat = assemble(a0).petscmat
    A1mat = assemble(a1).petscmat
    mats = [A0mat, A1mat]

    # create ensemble mat
    emat = EnsembleBlockDiagonalMat(mats, row_space, col_space)

    # build ensemble function lhs and rhs for Ax=y
    x = EnsembleFunction(row_space)
    y = EnsembleCofunction(col_space.dual())
    ycheck = EnsembleCofunction(col_space.dual())

    for i, xi in enumerate(x.subfunctions):
        xi.assign(ensemble_rank + i + 1)

    # assemble reference matmult
    for A, xi, yi in zip(mats, x.subfunctions, ycheck.subfunctions):
        with xi.dat.vec_ro as xv, yi.dat.vec_wo as yv:
            A.mult(xv, yv)

    # assemble matmult
    with x.vec_ro() as xv, y.vec_wo() as yv:
        emat.mult(xv, yv)

    checks = [
        np.allclose(yi.dat.data_ro, yci.dat.data_ro)
        for yi, yci in zip(y.subfunctions, ycheck.subfunctions)
    ]

    # check results
    parallel_assert(
        all(checks),
        msg=("Action of EnsembleBlockDiagonalMat does not match"
             f" actions of local matrices: {checks}")
    )


@pytest.mark.parallel([1, 2, 3, 4])
@pytest.mark.parametrize("default_options", [True, False],
                         ids=["default_options", "blockwise_options"])
def test_ensemble_pc(default_options):
    # create ensemble
    global_ranks = COMM_WORLD.size
    nspatial_ranks = 2 if (global_ranks % 2 == 0) else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)
    ensemble_rank = ensemble.ensemble_rank

    # Default PETSc pc is ILU so need a 2D mesh
    # because for 1D ILU is an exact solver.
    mesh = UnitSquareMesh(8, 8, comm=ensemble.comm)

    # create function spaces
    CG = FunctionSpace(mesh, "CG", 2)
    DG = FunctionSpace(mesh, "DG", 2+ensemble_rank)

    # create ensemble function spaces / functions
    row_space = EnsembleFunctionSpace([CG, DG], ensemble)
    col_space = EnsembleFunctionSpace([CG, DG], ensemble)
    offset = col_space.global_spaces_offset

    # build forms
    u, v = TrialFunction(CG), TestFunction(CG)
    nu = Constant(offset + 1)
    a0 = inner(u, v)*dx + nu*inner(grad(u), grad(v))*dx

    u, v = TrialFunction(DG), TestFunction(DG)
    a1 = (1/nu)*inner(u, v)*dx

    # assemble mats
    A0mat = assemble(a0, mat_type='aij').petscmat
    A1mat = assemble(a1, mat_type='aij').petscmat
    mats = [A0mat, A1mat]

    # create ensemble mat
    emat = EnsembleBlockDiagonalMat(mats, row_space, col_space)

    # parameters: direct solve on blocks
    parameters = {
        'ksp_rtol': 1e-14,
        'ksp_type': 'richardson',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.EnsembleBJacobiPC',
    }
    if default_options:
        parameters['sub_pc_type'] = 'lu'
    else:
        for i in range(col_space.nglobal_spaces):
            parameters[f'sub_{i}_pc_type'] = 'lu'

    # create ensemble ksp
    ksp = PETSc.KSP().create(comm=ensemble.global_comm)
    ksp.setOperators(emat, emat)
    petsctools.set_from_options(
        ksp, parameters=parameters,
        options_prefix="ensemble")

    x = EnsembleFunction(row_space)
    b = EnsembleFunction(col_space.dual())

    for i, bi in enumerate(b.subfunctions):
        bi.assign(offset + i + 1)

    with petsctools.inserted_options(ksp):
        with x.vec_wo() as xv, b.vec_ro() as bv:
            ksp.solve(bv, xv)

    # 1 richardson iteration should be a direct solve
    parallel_assert(
        ksp.its == 1,
        msg=("EnsembleBJacobiPC took more than one iteration to"
             f" solve an EnsembleBlockDiagonalMat: {ksp.its=}")
    )


if __name__ == "__main__":
    test_ensemble_pc(default_options=True)
