import pytest
from firedrake import *
from firedrake.cython import dmcommon


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


def _test_submesh_base_cell_integral_quad(family_degree, nelem):
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
    label_name = dmcommon.CELL_SETS_LABEL
    msub = Submesh(mesh, label_name, label_value, mesh.topological_dimension())
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
    family, degree = family_degree
    mesh = UnitSquareMesh(nelem, nelem, quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    cond = conditional(x > .5, 1,
           conditional(y > .5, 1, 0))  # noqa: E128
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(cond)
    label_value = 999
    mesh.mark_entities(indicator_function, label_value)
    label_name = dmcommon.CELL_SETS_LABEL
    subm = Submesh(mesh, label_name, label_value, mesh.topological_dimension())
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
    label_name = dmcommon.CELL_SETS_LABEL
    msub = Submesh(mesh, label_name, label_value, mesh.topological_dimension())
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


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('family_degree', [("Q", 4), ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_base_cell_integral_hex_3_processes(family_degree, nelem):
    _test_submesh_base_cell_integral_hex(family_degree, nelem)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('family_degree', [("Q", 4), ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_base_cell_integral_hex_4_processes(family_degree, nelem):
    _test_submesh_base_cell_integral_hex(family_degree, nelem)


def _test_submesh_base_facet_integral_hex(family_degree, nelem):
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
    label_name = dmcommon.CELL_SETS_LABEL
    subm = Submesh(mesh, label_name, label_value, mesh.topological_dimension())
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


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('family_degree', [("Q", 3), ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_base_facet_integral_hex_3_processes(family_degree, nelem):
    _test_submesh_base_facet_integral_hex(family_degree, nelem)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('family_degree', [("Q", 3), ])
@pytest.mark.parametrize('nelem', [2, 4, 8])
def test_submesh_base_facet_integral_hex_4_processes(family_degree, nelem):
    _test_submesh_base_facet_integral_hex(family_degree, nelem)
