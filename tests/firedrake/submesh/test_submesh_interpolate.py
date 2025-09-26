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
    subm = Submesh(mesh, dim, label_value)
    V = FunctionSpace(mesh, family, degree)
    V_ = FunctionSpace(mesh, family_sub, degree_sub)
    Vsub = FunctionSpace(subm, family_sub, degree_sub)
    Vsub_ = FunctionSpace(subm, family, degree)
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
    mesh_file = join(cwd, "..", "..", "..", "docs", "notebooks/stokes-control.msh")
    mesh = Mesh(mesh_file, distribution_parameters=distribution_parameters)
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


@pytest.mark.parallel(nprocs=5)
@pytest.mark.parametrize('hexahedral', [False, True])
@pytest.mark.parametrize('direction', ['x', 'y', 'z'])
@pytest.mark.parametrize('facet_type', ['interior', 'exterior'])
def test_submesh_interpolate_3Dcell_2Dfacet(hexahedral, direction, facet_type):
    def expr(m):
        x, y, z = SpatialCoordinate(m)
        return x + y**2 + z**3
    degree = 3
    distribution_parameters = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
    }
    if hexahedral:
        mesh = Mesh(
            join(cwd, "..", "meshes", "cube_hex.msh"),
            distribution_parameters=distribution_parameters,
        )
        V = FunctionSpace(mesh, "CG", 2)
    else:
        mesh = UnitCubeMesh(8, 8, 8)
        V = FunctionSpace(mesh, "HDiv Trace", 0)
    x, y, z = SpatialCoordinate(mesh)
    facet_function = Function(V).interpolate(
        conditional(
            {
                ('x', 'interior'): And(x > .499, x < .501),
                ('y', 'interior'): And(y > .499, y < .501),
                ('z', 'interior'): And(z > .499, z < .501),
                ('x', 'exterior'): x > .999,
                ('y', 'exterior'): y > .999,
                ('z', 'exterior'): z > .999,
            }[(direction, facet_type)],
            1., 0.,
        )
    )
    facet_value = 999
    mesh = RelabeledMesh(mesh, [facet_function], [facet_value])
    subm = Submesh(mesh, mesh.topological_dimension() - 1, facet_value)
    DG3d = FunctionSpace(mesh, "DG", degree)
    dg3d = Function(DG3d).interpolate(expr(mesh))
    DG2d = FunctionSpace(subm, "DG", degree)
    dg2d = Function(DG2d).interpolate(expr(subm))
    value3d_int = assemble(inner(dg3d('+'), dg3d('-')) * dS(facet_value))
    value3d_ext = assemble(inner(dg3d, dg3d) * ds(facet_value))
    value2d = assemble(inner(dg2d, dg2d) * dx)
    assert abs(value2d - (value3d_int + value3d_ext)) < 1.e-14
    if facet_type == 'exterior':
        x, y, z = SpatialCoordinate(subm)
        RT2d = FunctionSpace(subm, "RTCE" if hexahedral else "RTE", 4)
        tangent_expr = {
            'x': as_vector([0, y**2, z**3]),
            'y': as_vector([x, 0, z**3]),
            'z': as_vector([x, y**2, 0]),
        }[direction]
        rt2d = Function(RT2d).project(
            tangent_expr,
            solver_parameters={
                "ksp_rtol": 1.e-14,
            },
        )
        error_expr = rt2d - tangent_expr
        error = assemble(inner(error_expr, error_expr) * dx)**0.5
        assert abs(error) < 1.e-14


@pytest.mark.parallel(nprocs=4)
def test_submesh_interpolate_3Dcell_2Dfacet_simplex_sckelton():
    # The usage of sckelton meshes is limited as
    # number of support cells of a facet can be > 2.
    # We can not make sckelton mesh of hex meshes as,
    # already in the quad orientation implementation,
    # we assume that number of support cells <= 2.
    def expr(m):
        x, y, z = SpatialCoordinate(m)
        return x + y**2 + z**3
    degree = 3
    distribution_parameters = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
    }
    mesh = UnitCubeMesh(8, 8, 8, distribution_parameters=distribution_parameters)
    V = FunctionSpace(mesh, "HDiv Trace", 0)
    facet_function = Function(V).interpolate(Constant(1.))
    facet_value = 999
    mesh = RelabeledMesh(mesh, [facet_function], [facet_value])
    subm = Submesh(mesh, mesh.topological_dimension() - 1, facet_value)
    HDivT3d = FunctionSpace(mesh, "HDiv Trace", degree)
    hdivt3d = Function(HDivT3d).interpolate(expr(mesh))
    DG2d = FunctionSpace(subm, "DG", degree)
    dg2d = Function(DG2d).interpolate(expr(subm))
    value3d_int = assemble(inner(hdivt3d('+'), hdivt3d('-')) * dS(facet_value))
    value3d_ext = assemble(inner(hdivt3d, hdivt3d) * ds(facet_value))
    value2d = assemble(inner(dg2d, dg2d) * dx)
    assert abs(value2d - (value3d_int + value3d_ext)) < 5.e-13
    DG3d = FunctionSpace(mesh, "DG", degree)
    dg3d = Function(DG3d).interpolate(expr(mesh))
    dg2d_ = Function(DG2d).interpolate(dg3d)
    error = assemble(inner(dg2d_ - expr(subm), dg2d_ - expr(subm)) * dx)**0.5
    assert abs(error) < 1.e-14
