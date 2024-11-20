import pytest
from firedrake import *
import numpy as np
from ufl.conditional import GT, LT
from os.path import abspath, dirname, join


cwd = abspath(dirname(__file__))


def _get_expr(V):
    m = V.ufl_domain()
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
    if V.value_shape == ():
        return cos(x) + x * exp(y) + sin(z)
    elif V.value_shape == (2, ):
        return as_vector([cos(x), sin(y)])


def _test_submesh_interpolate_cell_cell(mesh, subdomain_cond, fe_fesub):
    dim = mesh.topological_dimension()
    (family, degree), (family_sub, degree_sub) = fe_fesub
    DG0 = FunctionSpace(mesh, "DG", 0)
    indicator_function = Function(DG0).interpolate(subdomain_cond)
    label_value = 999
    mesh.mark_entities(indicator_function, label_value)
    msub = Submesh(mesh, dim, label_value)
    V = FunctionSpace(mesh, family, degree)
    V_ = FunctionSpace(mesh, family_sub, degree_sub)
    Vsub = FunctionSpace(msub, family_sub, degree_sub)
    Vsub_ = FunctionSpace(msub, family, degree)
    f = Function(V).interpolate(_get_expr(V))
    gsub_ = Function(Vsub_).interpolate(_get_expr(Vsub_))
    gsub = Function(Vsub).interpolate(gsub_)
    fsub = Function(Vsub).interpolate(f)
    assert np.allclose(fsub.dat.data_ro_with_halos, gsub.dat.data_ro_with_halos)
    f = Function(V_).interpolate(f)
    g = Function(V)
    # interpolation on subdomain only makes sense
    # if there is no ambiguity on the subdomain boundary.
    # For testing, the following suffices.
    g.interpolate(f)
    temp = Constant(999.*np.ones(V.value_shape))
    g.interpolate(temp, subset=mesh.topology.cell_subset(label_value))  # pollute the data
    g.interpolate(gsub, subset=mesh.topology.cell_subset(label_value))
    assert assemble(inner(g - f, g - f) * dx(label_value)).real < 1e-14


@pytest.mark.parametrize('nelem', [2, 4, 8, None])
@pytest.mark.parametrize('fe_fesub', [[("DQ", 0), ("DQ", 0)],
                                      [("Q", 4), ("DQ", 5)]])
@pytest.mark.parametrize('condx', [LT])
@pytest.mark.parametrize('condy', [LT])
@pytest.mark.parametrize('condz', [LT])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_interpolate_cell_cell_hex_1_processes(fe_fesub, nelem, condx, condy, condz, distribution_parameters):
    if nelem is None:
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"), distribution_parameters=distribution_parameters)
    else:
        mesh = UnitCubeMesh(nelem, nelem, nelem, hexahedral=True, distribution_parameters=distribution_parameters)
    x, y, z = SpatialCoordinate(mesh)
    cond = conditional(condx(x, .5), 1,
           conditional(condy(y, .5), 1,       # noqa: E128
           conditional(condz(z, .5), 1, 0)))  # noqa: E128
    _test_submesh_interpolate_cell_cell(mesh, cond, fe_fesub)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('nelem', [2, 4, 8, None])
@pytest.mark.parametrize('fe_fesub', [[("DQ", 0), ("DQ", 0)],
                                      [("Q", 4), ("DQ", 5)]])
@pytest.mark.parametrize('condx', [LT, GT])
@pytest.mark.parametrize('condy', [LT, GT])
@pytest.mark.parametrize('condz', [LT, GT])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_interpolate_cell_cell_hex_3_processes(fe_fesub, nelem, condx, condy, condz, distribution_parameters):
    if nelem is None:
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"), distribution_parameters=distribution_parameters)
    else:
        mesh = UnitCubeMesh(nelem, nelem, nelem, hexahedral=True, distribution_parameters=distribution_parameters)
    x, y, z = SpatialCoordinate(mesh)
    cond = conditional(condx(x, .5), 1,
           conditional(condy(y, .5), 1,       # noqa: E128
           conditional(condz(z, .5), 1, 0)))  # noqa: E128
    _test_submesh_interpolate_cell_cell(mesh, cond, fe_fesub)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('fe_fesub', [[("DP", 0), ("DP", 0)],
                                      [("P", 4), ("DP", 5)],
                                      [("BDME", 2), ("BDME", 3)],
                                      [("BDMF", 2), ("BDMF", 3)]])
@pytest.mark.parametrize('condx', [LT, GT])
@pytest.mark.parametrize('condy', [LT, GT])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_interpolate_cell_cell_tri_3_processes(fe_fesub, condx, condy, distribution_parameters):
    mesh = Mesh("./docs/notebooks/stokes-control.msh", distribution_parameters=distribution_parameters)
    x, y = SpatialCoordinate(mesh)
    cond = conditional(condx(x, 15.), 1,
           conditional(condy(y, 2.5), 1, 0))  # noqa: E128
    _test_submesh_interpolate_cell_cell(mesh, cond, fe_fesub)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('fe_fesub', [[("DQ", 0), ("DQ", 0)],
                                      [("Q", 4), ("DQ", 5)]])
@pytest.mark.parametrize('condx', [LT, GT])
@pytest.mark.parametrize('condy', [LT, GT])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_interpolate_cell_cell_quad_3_processes(fe_fesub, condx, condy, distribution_parameters):
    mesh = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"), distribution_parameters=distribution_parameters)
    x, y = SpatialCoordinate(mesh)
    cond = conditional(condx(x, 0.5), 1,
           conditional(condy(y, 0.5), 1, 0))  # noqa: E128
    _test_submesh_interpolate_cell_cell(mesh, cond, fe_fesub)
