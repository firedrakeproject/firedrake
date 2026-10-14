import pytest
from firedrake import *
import numpy as np


def test_single_mesh_mixed_assign():
    """Assigning between functions on separately constructed but equivalent
    MixedFunctionSpaces should work and preserve values."""
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "CG", 1)

    z = Function(MixedFunctionSpace([V, W]))
    z.subfunctions[0].assign(Constant((1.0, 2.0)))
    z.subfunctions[1].assign(3.0)

    w = Function(MixedFunctionSpace([V, W]))
    w.assign(z)

    assert np.allclose(w.subfunctions[0].dat.data_ro, [1.0, 2.0])
    assert np.allclose(w.subfunctions[1].dat.data_ro, 3.0)


@pytest.mark.parallel(nprocs=[1, 3])
def test_assign_cell_subset():
    """Assigning over a subset of the cells writes the nodes of those cells."""
    sentinel = -1.0
    mesh = UnitSquareMesh(6, 6)
    x, y = SpatialCoordinate(mesh)
    marker = Function(FunctionSpace(mesh, "DG", 0)).interpolate(conditional(x < 0.5, 1, 0))
    mesh.mark_entities(marker, 7)

    V = FunctionSpace(mesh, "CG", 2)
    source = Function(V).interpolate(sin(3 * x))
    target = Function(V).assign(sentinel)
    target.assign(source, subset=mesh.cell_subset(7))

    written = target.dat.data_ro != sentinel
    assert np.allclose(target.dat.data_ro[written], source.dat.data_ro[written])
    assert np.all(target.dat.data_ro[~written] == sentinel)
    # The nodes of the unmarked cells are left alone, except where they sit on
    # a cell of the subset too.
    coords = Function(VectorFunctionSpace(mesh, "CG", 2)).interpolate(SpatialCoordinate(mesh))
    assert not written[coords.dat.data_ro[:, 0] > 0.5 + 1e-12].any()
    # Which nodes the subset holds cannot depend on how the mesh is
    # partitioned, so the count is the same however many ranks there are.
    nwritten = written[:V.dof_dset.size]
    assert mesh.comm.allreduce(int(nwritten.sum())) == 91
