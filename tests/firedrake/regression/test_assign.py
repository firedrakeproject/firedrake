from firedrake import *
import numpy as np
import pytest


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("extruded", (False, True))
@pytest.mark.parametrize("into_restricted", (False, True))
@pytest.mark.parametrize("subset_only", (False, True))
@pytest.mark.parametrize("valid_halos", (False, True))
def test_restricted_assignment_ordering(extruded, into_restricted, subset_only, valid_halos):
    """Reordered assignment includes all layers, components, and halo values."""
    mesh = UnitSquareMesh(2, 2, quadrilateral=True)
    if extruded:
        mesh = ExtrudedMesh(mesh, 3)
    V = VectorFunctionSpace(mesh, "Q", 2, dim=2)
    W = RestrictedFunctionSpace(V, ("on_boundary",))
    source_space, target_space = (V, W) if into_restricted else (W, V)
    x = SpatialCoordinate(mesh)
    expression = as_vector((x[0] + 3 * x[mesh.geometric_dimension - 1], x[1] - 2 * x[0]))
    source = Function(source_space).interpolate(expression)
    if valid_halos:
        _ = source.dat.data_ro_with_halos
    else:
        _ = source.dat.data
    target = Function(target_space)
    expected = Function(target_space).interpolate(expression)
    subset = DirichletBC(target_space, 0, "on_boundary").node_set if subset_only else None
    target.assign(source, subset=subset)
    expected_values = expected.dat.data_ro_with_halos.copy()
    if subset_only:
        mask = np.ones(target_space.node_set.total_size, dtype=bool)
        mask[subset.indices] = False
        expected_values[mask] = 0
    assert np.allclose(target.dat.data_ro_with_halos, expected_values)


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
