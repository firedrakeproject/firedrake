import pytest
from firedrake import *


def _get_expr(m):
    if m.geometric_dimension == 1:
        x, = SpatialCoordinate(m)
        y = x * x
        z = x + y
    elif m.geometric_dimension == 2:
        x, y = SpatialCoordinate(m)
        z = x + y
    elif m.geometric_dimension == 3:
        x, y, z = SpatialCoordinate(m)
    else:
        raise NotImplementedError("Not implemented")
    return exp(x + y * y + z * z * z)


def _test_submesh_base_cell_integral_quad(family_degree, nelem):
    dim = 2
    family, degree = family_degree
    mesh = UnitSquareMesh(nelem, nelem, quadrilateral=True)
    V = FunctionSpace(mesh, family, degree)
    f = Function(V).interpolate(_get_expr(mesh))
    x, y = SpatialCoordinate(mesh)
    cond = conditional(x > .5, 1,
           conditional(y > .5, 1, 0))  # noqa: E128
    target = assemble(f * cond * dx)
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(cond)
    label_value = 999
    mesh.mark_entities(indicator_function, label_value)
    msub = Submesh(mesh, dim, label_value)
    Vsub = FunctionSpace(msub, family, degree)
    fsub = Function(Vsub).interpolate(_get_expr(msub))
    result = assemble(fsub * dx)
    assert abs(result - target) < 1e-12


@pytest.mark.parametrize('family_degree', [("Q", 4), ])
@pytest.mark.parametrize('nelem', [2, 4, 8, 16])
def test_submesh_base_cell_integral_quad_1_process(family_degree, nelem):
    _test_submesh_base_cell_integral_quad(family_degree, nelem)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('family_degree', [("Q", 4), ])
@pytest.mark.parametrize('nelem', [2, 4, 8, 16])
def test_submesh_base_cell_integral_quad_2_processes(family_degree, nelem):
    _test_submesh_base_cell_integral_quad(family_degree, nelem)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('family_degree', [("Q", 4), ])
@pytest.mark.parametrize('nelem', [2, 4, 8, 16])
def test_submesh_base_cell_integral_quad_3_processes(family_degree, nelem):
    _test_submesh_base_cell_integral_quad(family_degree, nelem)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('family_degree', [("Q", 4), ])
@pytest.mark.parametrize('nelem', [2, 4, 8, 16])
def test_submesh_base_cell_integral_quad_4_processes(family_degree, nelem):
    _test_submesh_base_cell_integral_quad(family_degree, nelem)


def _test_submesh_base_facet_integral_quad(family_degree, nelem):
    dim = 2
    family, degree = family_degree
    mesh = UnitSquareMesh(nelem, nelem, quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    cond = conditional(x > .5, 1,
           conditional(y > .5, 1, 0))  # noqa: E128
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(cond)
    label_value = 999
    mesh.mark_entities(indicator_function, label_value)
    subm = Submesh(mesh, dim, label_value)
    for i in [1, 2, 3, 4]:
        target = assemble(cond * _get_expr(mesh) * ds(i))
        result = assemble(_get_expr(subm) * ds(i))
        assert abs(result - target) < 2e-12
    # Check new boundary.
    assert abs(assemble(Constant(1.) * ds(subdomain_id=5, domain=subm)) - 1.0) < 1e-12
    x, y = SpatialCoordinate(subm)
    assert abs(assemble(x**4 * ds(5)) - (.5**5 / 5 + .5**4 * .5)) < 1e-12
    assert abs(assemble(y**4 * ds(5)) - (.5**5 / 5 + .5**4 * .5)) < 1e-12


@pytest.mark.parametrize('family_degree', [("Q", 3), ])
@pytest.mark.parametrize('nelem', [2, 4, 8, 16])
def test_submesh_base_facet_integral_quad_1_process(family_degree, nelem):
    _test_submesh_base_facet_integral_quad(family_degree, nelem)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('family_degree', [("Q", 3), ])
@pytest.mark.parametrize('nelem', [2, 4, 8, 16])
def test_submesh_base_facet_integral_quad_2_processes(family_degree, nelem):
    _test_submesh_base_facet_integral_quad(family_degree, nelem)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('family_degree', [("Q", 3), ])
@pytest.mark.parametrize('nelem', [2, 4, 8, 16])
def test_submesh_base_facet_integral_quad_3_processes(family_degree, nelem):
    _test_submesh_base_facet_integral_quad(family_degree, nelem)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('family_degree', [("Q", 3), ])
@pytest.mark.parametrize('nelem', [2, 4, 8, 16])
def test_submesh_base_facet_integral_quad_4_processes(family_degree, nelem):
    _test_submesh_base_facet_integral_quad(family_degree, nelem)


def _test_submesh_base_cell_integral_hex(family_degree, nelem):
    dim = 3
    family, degree = family_degree
    mesh = UnitCubeMesh(nelem, nelem, nelem, hexahedral=True)
    V = FunctionSpace(mesh, family, degree)
    f = Function(V).interpolate(_get_expr(mesh))
    x, y, z = SpatialCoordinate(mesh)
    cond = conditional(x > .5, 1,
           conditional(y > .5, 1,       # noqa: E128
           conditional(z > .5, 1, 0)))  # noqa: E128
    target = assemble(f * cond * dx)
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(cond)
    label_value = 999
    mesh.mark_entities(indicator_function, label_value)
    msub = Submesh(mesh, dim, label_value)
    Vsub = FunctionSpace(msub, family, degree)
    fsub = Function(Vsub).interpolate(_get_expr(msub))
    result = assemble(fsub * dx)
    assert abs(result - target) < 1e-12


@pytest.mark.parametrize('family_degree', [("Q", 4), ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_base_cell_integral_hex_1_process(family_degree, nelem):
    _test_submesh_base_cell_integral_hex(family_degree, nelem)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('family_degree', [("Q", 4), ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_base_cell_integral_hex_2_processes(family_degree, nelem):
    _test_submesh_base_cell_integral_hex(family_degree, nelem)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('family_degree', [("Q", 4), ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_base_cell_integral_hex_4_processes(family_degree, nelem):
    _test_submesh_base_cell_integral_hex(family_degree, nelem)


def _test_submesh_base_facet_integral_hex(family_degree, nelem):
    dim = 3
    family, degree = family_degree
    mesh = UnitCubeMesh(nelem, nelem, nelem, hexahedral=True)
    x, y, z = SpatialCoordinate(mesh)
    cond = conditional(x > .5, 1,
           conditional(y > .5, 1,       # noqa: E128
           conditional(z > .5, 1, 0)))  # noqa: E128
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(cond)
    label_value = 999
    mesh.mark_entities(indicator_function, label_value)
    subm = Submesh(mesh, dim, label_value)
    for i in [1, 2, 3, 4, 5, 6]:
        target = assemble(cond * _get_expr(mesh) * ds(i))
        result = assemble(_get_expr(subm) * ds(i))
        assert abs(result - target) < 2e-12
    # Check new boundary.
    assert abs(assemble(Constant(1) * ds(subdomain_id=7, domain=subm)) - .75) < 1e-12
    x, y, z = SpatialCoordinate(subm)
    assert abs(assemble(x**4 * ds(7)) - (.5**5 / 5 * .5 * 2 + .5**4 * .5**2)) < 1e-12
    assert abs(assemble(y**4 * ds(7)) - (.5**5 / 5 * .5 * 2 + .5**4 * .5**2)) < 1e-12
    assert abs(assemble(z**4 * ds(7)) - (.5**5 / 5 * .5 * 2 + .5**4 * .5**2)) < 1e-12


@pytest.mark.parametrize('family_degree', [("Q", 3), ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_base_facet_integral_hex_1_process(family_degree, nelem):
    _test_submesh_base_facet_integral_hex(family_degree, nelem)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('family_degree', [("Q", 3), ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_base_facet_integral_hex_2_processes(family_degree, nelem):
    _test_submesh_base_facet_integral_hex(family_degree, nelem)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('family_degree', [("Q", 3), ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_base_facet_integral_hex_4_processes(family_degree, nelem):
    _test_submesh_base_facet_integral_hex(family_degree, nelem)


@pytest.mark.parallel(nprocs=2)
def test_submesh_base_entity_maps():

    #  3---9--(5)-(12)(7)    (7)-(13)-3---9---5
    #  |       |       |      |       |       |
    #  8   0 (11) (1) (13)  (12) (1)  8   0  10  mesh
    #  |       |       |      |       |       |
    #  2--10--(4)(14)-(6)    (6)-(14)-2--11---4
    #
    #  2---6---4             (4)-(7)-(2)
    #  |       |              |       |
    #  5   0   8             (6) (0) (5)         submesh
    #  |       |              |       |
    #  1---7---3             (3)-(8)-(1)
    #
    #      rank 0                 rank 1

    dim = 2
    mesh = RectangleMesh(2, 1, 2., 1., quadrilateral=True, distribution_parameters={"partitioner_type": "simple"})
    assert mesh.comm.size == 2
    rank = mesh.comm.rank
    x, y = SpatialCoordinate(mesh)
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(conditional(x < 1., 1, 0))
    label_value = 999
    mesh.mark_entities(indicator_function, label_value)
    submesh = Submesh(mesh, dim, label_value)
    submesh.topology_dm.viewFromOptions("-dm_view")
    subdm = submesh.topology.topology_dm
    if rank == 0:
        assert subdm.getLabel("pyop2_core").getStratumSize(1) == 0
        assert subdm.getLabel("pyop2_owned").getStratumSize(1) == 9
        assert subdm.getLabel("pyop2_ghost").getStratumSize(1) == 0
        assert (subdm.getLabel("pyop2_owned").getStratumIS(1).getIndices() == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
        assert (mesh.interior_facets.facets == np.array([11])).all
        assert (mesh.exterior_facets.facets == np.array([8, 9, 10, 12, 13, 14])).all
        assert (submesh.interior_facets.facets == np.array([])).all
        assert (submesh.exterior_facets.facets == np.array([5, 6, 8, 7])).all()
    else:
        assert subdm.getLabel("pyop2_core").getStratumSize(1) == 0
        assert subdm.getLabel("pyop2_owned").getStratumSize(1) == 0
        assert subdm.getLabel("pyop2_ghost").getStratumSize(1) == 9
        assert (subdm.getLabel("pyop2_ghost").getStratumIS(1).getIndices() == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()
        assert (mesh.interior_facets.facets == np.array([8])).all
        assert (mesh.exterior_facets.facets == np.array([9, 10, 11, 12, 13, 14])).all
        assert (submesh.interior_facets.facets == np.array([])).all
        assert (submesh.exterior_facets.facets == np.array([6, 7, 5, 8])).all()
    composed_map, integral_type = mesh.topology.trans_mesh_entity_map(submesh.topology, "cell", None, None)
    assert integral_type == "cell"
    if rank == 0:
        assert (composed_map.maps_[0].values_with_halo == np.array([0])).all()
    else:
        assert (composed_map.maps_[0].values_with_halo == np.array([1])).all()
    composed_map, integral_type = mesh.topology.trans_mesh_entity_map(submesh.topology, "exterior_facet", 5, None)
    assert integral_type == "interior_facet"
    if rank == 0:
        assert (composed_map.maps_[0].values_with_halo == np.array([-1, -1, 0, -1]).reshape((-1, 1))).all()  # entire exterior-interior map
    else:
        assert (composed_map.maps_[0].values_with_halo == np.array([-1, -1, 0, -1]).reshape((-1, 1))).all()  # entire exterior-interior map
    composed_map, integral_type = mesh.topology.trans_mesh_entity_map(submesh.topology, "exterior_facet", 4, None)
    assert integral_type == "exterior_facet"
    if rank == 0:
        assert (composed_map.maps_[0].values_with_halo == np.array([0, 1, -1, 2]).reshape((-1, 1))).all()  # entire exterior-exterior map
    else:
        assert (composed_map.maps_[0].values_with_halo == np.array([3, 4, -1, 5]).reshape((-1, 1))).all()  # entire exterior-exterior map
    composed_map, integral_type = submesh.topology.trans_mesh_entity_map(mesh.topology, "exterior_facet", 1, None)
    assert integral_type == "exterior_facet"
    if rank == 0:
        assert (composed_map.maps_[0].values_with_halo == np.array([0, 1, 3, -1, -1, -1]).reshape((-1, 1))).all()
    else:
        assert (composed_map.maps_[0].values_with_halo == np.array([-1, -1, -1, 0, 1, 3]).reshape((-1, 1))).all()
