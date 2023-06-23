import pytest
from firedrake import *
from firedrake.cython import dmcommon
import numpy as np
from ufl.conditional import GT, LT


def _get_expr(m):
    if m.geometric_dimension() == 1:
        x, = SpatialCoordinate(m)
        y = x * x
        z = x + y
    elif m.geometric_dimension() == 2:
        x, y = SpatialCoordinate(m)
        z = x + y
    elif m.geometric_dimension() == 3:
        x, y, z = SpatialCoordinate(m)
    else:
        raise NotImplementedError("Not implemented")
    return exp(x + y * y + z * z * z)


def _test_submesh_interpolate_cell_cell_quad(family_degree, nelem):
    family, degree = family_degree
    mesh = UnitSquareMesh(nelem, nelem, quadrilateral=True)
    V = FunctionSpace(mesh, family, degree)
    f = Function(V).interpolate(_get_expr(mesh))
    x, y = SpatialCoordinate(mesh)
    cond = conditional(x > .5, 1,
           conditional(y < .5, 1, 0))  # noqa: E128
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(cond)
    label_name = dmcommon.CELL_SETS_LABEL
    label_value = 999
    mesh.mark_entities(indicator_function, label_name, label_value)
    msub = Submesh(mesh, label_name, label_value, mesh.topological_dimension())
    Vsub = FunctionSpace(msub, family, degree)
    fsub = Function(Vsub).interpolate(f)
    g = Function(Vsub).interpolate(_get_expr(msub))
    assert np.allclose(fsub.dat.data_ro_with_halos, g.dat.data_ro_with_halos)


@pytest.mark.parametrize('family_degree', [("DQ", 0),
                                           # ("Q", 4),  # Fails due to how cell_node_map is defined on quad meshes
                                          ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_interpolate_cell_cell_quad_1_processes(family_degree, nelem):
    _test_submesh_interpolate_cell_cell_quad(family_degree, nelem)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('family_degree', [("DQ", 0),
                                           # ("DQ", 1),  # Fails due to how cell_node_map is defined on quad meshes
                                           # ("Q", 4),  # Fails due to how cell_node_map is defined on quad meshes
                                          ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_interpolate_cell_cell_quad_4_processes(family_degree, nelem):
    _test_submesh_interpolate_cell_cell_quad(family_degree, nelem)


def _test_submesh_interpolate_cell_cell_hex(family_degree_pair, nelem, condx, condy, condz, distribution_parameters):
    (family, degree), (family_sub, degree_sub) = family_degree_pair
    mesh = UnitCubeMesh(nelem, nelem, nelem, hexahedral=True, distribution_parameters=distribution_parameters)
    x, y, z = SpatialCoordinate(mesh)
    cond = conditional(condx(x, .5), 1,
           conditional(condy(y, .5), 1,       # noqa: E128
           conditional(condz(z, .5), 1, 0)))  # noqa: E128
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(cond)
    label_name = dmcommon.CELL_SETS_LABEL
    label_value = 999
    mesh.mark_entities(indicator_function, label_name, label_value)
    msub = Submesh(mesh, label_name, label_value, mesh.topological_dimension())
    V = FunctionSpace(mesh, family, degree)
    f = Function(V).interpolate(_get_expr(mesh))
    Vsub = FunctionSpace(msub, family_sub, degree_sub)
    fsub = Function(Vsub).interpolate(f)
    Vsub_ = FunctionSpace(msub, family, degree)
    gsub_ = Function(Vsub_).interpolate(_get_expr(msub))
    gsub = Function(Vsub).interpolate(gsub_)
    assert np.allclose(fsub.dat.data_ro_with_halos, gsub.dat.data_ro_with_halos)
    g = Function(V).interpolate(gsub, subset=mesh.topology.cell_subset(label_value))
    assert assemble((g - f) ** 2 * dx(label_value)).real < 1e-14


@pytest.mark.parametrize('family_degree_pair', [[("DQ", 0), ("DQ", 0)],
                                                [("Q", 4), ("DQ", 5)]])
@pytest.mark.parametrize('nelem', [2, 4, 8])
@pytest.mark.parametrize('condx', [LT])
@pytest.mark.parametrize('condy', [LT])
@pytest.mark.parametrize('condz', [LT])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_interpolate_cell_cell_hex_1_processes(family_degree_pair, nelem, condx, condy, condz, distribution_parameters):
    _test_submesh_interpolate_cell_cell_hex(family_degree_pair, nelem, condx, condy, condz, distribution_parameters)


@pytest.mark.parallel(nprocs=8)
@pytest.mark.parametrize('family_degree_pair', [[("DQ", 0), ("DQ", 0)],
                                                [("Q", 4), ("DQ", 5)]])
@pytest.mark.parametrize('nelem', [2, 4, 8])
@pytest.mark.parametrize('condx', [LT, GT])
@pytest.mark.parametrize('condy', [LT, GT])
@pytest.mark.parametrize('condz', [LT, GT])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_interpolate_cell_cell_hex_8_processes(family_degree_pair, nelem, condx, condy, condz, distribution_parameters):
    _test_submesh_interpolate_cell_cell_hex(family_degree_pair, nelem, condx, condy, condz, distribution_parameters)
