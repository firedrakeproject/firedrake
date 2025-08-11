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


def _test_submesh_assign_cell_cell(mesh, subdomain_cond, fe_fesub):
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
    v0 = Coargument(V.dual(), 0)
    v1 = TrialFunction(Vsub)
    interp = Interpolate(v1, v0, allow_missing_dofs=True)
    A = assemble(interp)
    g = assemble(action(A, gsub))
    assert assemble(inner(g - f, g - f) * dx(label_value)).real < 1e-14


@pytest.mark.parametrize('nelem', [2, 4, 8, None])
@pytest.mark.parametrize('fe_fesub', [[("DQ", 0), ("DQ", 0)],
                                      [("Q", 4), ("Q", 5)]])
@pytest.mark.parametrize('condx', [LT])
@pytest.mark.parametrize('condy', [LT])
@pytest.mark.parametrize('condz', [LT])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_assign_cell_cell_hex_1_processes(fe_fesub, nelem, condx, condy, condz, distribution_parameters):
    if nelem is None:
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"), distribution_parameters=distribution_parameters)
    else:
        mesh = UnitCubeMesh(nelem, nelem, nelem, hexahedral=True, distribution_parameters=distribution_parameters)
    x, y, z = SpatialCoordinate(mesh)
    cond = conditional(condx(x, .5), 1,
           conditional(condy(y, .5), 1,       # noqa: E128
           conditional(condz(z, .5), 1, 0)))  # noqa: E128
    _test_submesh_assign_cell_cell(mesh, cond, fe_fesub)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('nelem', [2, 4, 8, None])
@pytest.mark.parametrize('fe_fesub', [[("DQ", 0), ("DQ", 0)],
                                      [("Q", 4), ("Q", 5)]])
@pytest.mark.parametrize('condx', [LT, GT])
@pytest.mark.parametrize('condy', [LT, GT])
@pytest.mark.parametrize('condz', [LT, GT])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_assign_cell_cell_hex_3_processes(fe_fesub, nelem, condx, condy, condz, distribution_parameters):
    if nelem is None:
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"), distribution_parameters=distribution_parameters)
    else:
        mesh = UnitCubeMesh(nelem, nelem, nelem, hexahedral=True, distribution_parameters=distribution_parameters)
    x, y, z = SpatialCoordinate(mesh)
    cond = conditional(condx(x, .5), 1,
           conditional(condy(y, .5), 1,       # noqa: E128
           conditional(condz(z, .5), 1, 0)))  # noqa: E128
    _test_submesh_assign_cell_cell(mesh, cond, fe_fesub)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('fe_fesub', [[("DP", 0), ("DP", 0)],
                                      [("P", 4), ("P", 5)],
                                      [("BDME", 2), ("BDME", 3)],
                                      [("BDMF", 2), ("BDMF", 3)]])
@pytest.mark.parametrize('condx', [LT, GT])
@pytest.mark.parametrize('condy', [LT, GT])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_assign_cell_cell_tri_3_processes(fe_fesub, condx, condy, distribution_parameters):
    mesh_file = join(cwd, "..", "..", "..", "docs", "notebooks/stokes-control.msh")
    mesh = Mesh(mesh_file, distribution_parameters=distribution_parameters)
    x, y = SpatialCoordinate(mesh)
    cond = conditional(condx(x, 15.), 1,
           conditional(condy(y, 2.5), 1, 0))  # noqa: E128
    _test_submesh_assign_cell_cell(mesh, cond, fe_fesub)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('fe_fesub', [[("DQ", 0), ("DQ", 0)],
                                      [("Q", 4), ("Q", 5)]])
@pytest.mark.parametrize('condx', [LT, GT])
@pytest.mark.parametrize('condy', [LT, GT])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_assign_cell_cell_quad_3_processes(fe_fesub, condx, condy, distribution_parameters):
    mesh = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"), distribution_parameters=distribution_parameters)
    x, y = SpatialCoordinate(mesh)
    cond = conditional(condx(x, 0.5), 1,
           conditional(condy(y, 0.5), 1, 0))  # noqa: E128
    _test_submesh_assign_cell_cell(mesh, cond, fe_fesub)


@pytest.mark.parallel(nprocs=2)
def test_submesh_assign_3_cell_2_processes():
    # mesh
    # rank 0:
    # 4---12----6---15---(8)-(18)-(10)
    # |         |         |         |
    # 11   0   13    1  (17)  (2) (19)
    # |         |         |         |
    # 3---14----5---16---(7)-(20)--(9)
    # rank 1:
    #          (7)-(13)---3----9----5
    #           |         |         |
    #          (12) (1)   8    0   10
    #           |         |         |    plex points
    #          (6)-(14)---2---11----4    () = ghost
    left = 111
    right = 222
    middle = 111222
    mesh = RectangleMesh(
        3, 1, 3., 1., quadrilateral=True, distribution_parameters={"partitioner_type": "simple"},
    )
    dim = mesh.topological_dimension()
    x, _ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_l = Function(DG0).interpolate(conditional(x < 2.0, 1, 0))
    f_r = Function(DG0).interpolate(conditional(x > 1.0, 1, 0))
    f_m = Function(DG0).interpolate(conditional(And(x < 2.0, x > 1.0), 1, 0))
    mesh = RelabeledMesh(mesh, [f_l, f_r, f_m], [left, right, middle])
    mesh_l = Submesh(mesh, dim, left)
    mesh_r = Submesh(mesh, dim, right)
    V = VectorFunctionSpace(mesh, "CG", 1)
    V_l = VectorFunctionSpace(mesh_l, "CG", 1)
    V_r = VectorFunctionSpace(mesh_r, "CG", 1)
    # Test various combinations.
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_l.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(left)))
    assert(abs(e) < 1.e-15)
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_r.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(right)))
    assert(abs(e) < 1.e-15)
    x = SpatialCoordinate(mesh_l)
    f = Function(V_l).assign(mesh.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(left)))
    assert(abs(e) < 1.e-15)
    x = SpatialCoordinate(mesh_r)
    f = Function(V_r).assign(mesh.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(right)))
    assert(abs(e) < 1.e-15)
    x = SpatialCoordinate(mesh_r)
    f = Function(V_r).assign(mesh_l.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(middle)))
    assert(abs(e) < 1.e-15)
    x = SpatialCoordinate(mesh_l)
    f = Function(V_l).assign(mesh_r.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(middle)))
    assert(abs(e) < 1.e-15)


@pytest.mark.parallel(nprocs=2)
def test_submesh_assign_2_cell_2_processes_no_overlap():
    # mesh
    # rank 0:
    # 2----6---(4)
    # |         |
    # 5    0   (7)
    # |         |
    # 1----8---(3)
    # rank 1:
    #           2----6----4
    #           |         |
    #           5    0    7
    #           |         |    plex points
    #           1----8----3    () = ghost
    left = 111
    right = 222
    distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.NONE, 0),
        "partitioner_type": "simple",
    }
    mesh = RectangleMesh(
        2, 1, 2., 1., quadrilateral=True, distribution_parameters=distribution_parameters,
    )
    dim = mesh.topological_dimension()
    x, _ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_l = Function(DG0).interpolate(conditional(x < 1.0, 1, 0))
    f_r = Function(DG0).interpolate(conditional(x > 1.0, 1, 0))
    mesh = RelabeledMesh(mesh, [f_l, f_r], [left, right])
    mesh_l = Submesh(mesh, dim, left)
    mesh_r = Submesh(mesh, dim, right)
    V = VectorFunctionSpace(mesh, "CG", 1)
    V_l = VectorFunctionSpace(mesh_l, "CG", 1)
    V_r = VectorFunctionSpace(mesh_r, "CG", 1)
    # Test various combinations.
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_r.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(right)))
    assert(abs(e) < 1.e-15)
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_l.coordinates)
    print(mesh.comm.rank)
    print(f.dat._data)
    print(f.dat.data_ro_with_halos)
    e = sqrt(assemble(inner(f - x, f - x) * dx(left)))
    assert(abs(e) < 1.e-15)
