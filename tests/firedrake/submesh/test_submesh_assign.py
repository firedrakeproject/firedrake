import pytest
import numpy as np
from firedrake import *
import finat
from functools import partial
from os.path import abspath, dirname, join


cwd = abspath(dirname(__file__))


@pytest.mark.parallel(nprocs=2)
def test_submesh_assign_function_3_quads_2_processes():
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
    dim = mesh.topological_dimension
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
    f = Function(V).assign(mesh_l.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(left)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_r.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(right)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh_l)
    f = Function(V_l).assign(mesh.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(left)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh_r)
    f = Function(V_r).assign(mesh.coordinates)
    e = sqrt(assemble(inner(f - x, f - x) * dx(right)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh_r)
    f = Function(V_r).assign(mesh_l.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(middle)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh_l)
    f = Function(V_l).assign(mesh_r.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(middle)))
    assert abs(e) < 1.e-15


@pytest.mark.parallel(nprocs=2)
def test_submesh_assign_function_2_quads_2_processes_no_overlap():
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
    distribution_parameters = {
        "overlap_type": (DistributedMeshOverlapType.NONE, 0),
        "partitioner_type": "simple",
    }
    mesh = RectangleMesh(
        2, 1, 2., 1., quadrilateral=True, distribution_parameters=distribution_parameters,
    )
    dim = mesh.topological_dimension
    x, _ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_l = Function(DG0).interpolate(conditional(x < 1.0, 1, 0))
    f_r = Function(DG0).interpolate(conditional(x > 1.0, 1, 0))
    mesh = RelabeledMesh(mesh, [f_l, f_r], [left, right])
    mesh_l = Submesh(mesh, dim, left)
    mesh_r = Submesh(mesh, dim, right)
    elem = mesh.ufl_coordinate_element()
    V = FunctionSpace(mesh, elem)
    # Test various combinations.
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_r.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(right)))
    assert abs(e) < 1.e-15
    x = SpatialCoordinate(mesh)
    f = Function(V).assign(mesh_l.coordinates, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f - x, f - x) * dx(left)))
    assert abs(e) < 1.e-15


@pytest.mark.parallel(nprocs=8)
@pytest.mark.parametrize('simplex', [True, False])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_assign_function_unstructured_8_processes(simplex, distribution_parameters):
    if not simplex and distribution_parameters == {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}:
        pytest.skip(reason="quad orientation bug; see https://github.com/firedrakeproject/firedrake/issues/4476")
    left = 111
    right = 222
    middle = 111222
    if simplex:
        mesh_file = join(cwd, "..", "..", "..", "docs", "notebooks/stokes-control.msh")
        mesh = Mesh(mesh_file, distribution_parameters=distribution_parameters)
        x, _ = SpatialCoordinate(mesh)
        DG0 = FunctionSpace(mesh, "DP", 0)
        f_l = Function(DG0).interpolate(conditional(x < 15., 1, 0))
        f_r = Function(DG0).interpolate(conditional(x > 7., 1, 0))
        f_m = Function(DG0).interpolate(conditional(And(x < 15., x > 7.), 1, 0))
        mesh = RelabeledMesh(mesh, [f_l, f_r, f_m], [left, right, middle])
        elem = finat.ufl.FiniteElement("RT", mesh.ufl_cell(), 2)
    else:
        mesh = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"), distribution_parameters=distribution_parameters)
        x, _ = SpatialCoordinate(mesh)
        DG0 = FunctionSpace(mesh, "DQ", 0)
        f_l = Function(DG0).interpolate(conditional(x < .75, 1, 0))
        f_r = Function(DG0).interpolate(conditional(x > .50, 1, 0))
        f_m = Function(DG0).interpolate(conditional(And(x < .75, x > .50), 1, 0))
        mesh = RelabeledMesh(mesh, [f_l, f_r, f_m], [left, right, middle])
        elem = finat.ufl.FiniteElement("RTCF", mesh.ufl_cell(), 2)
    dim = mesh.topological_dimension
    mesh_l = Submesh(mesh, dim, left)
    mesh_r = Submesh(mesh, dim, right)
    V = FunctionSpace(mesh, elem)
    V_l = FunctionSpace(mesh_l, elem)
    V_r = FunctionSpace(mesh_r, elem)
    f = Function(V).project(mesh.coordinates, solver_parameters={"ksp_rtol": 1.e-16})
    f_l = Function(V_l).project(mesh_l.coordinates, solver_parameters={"ksp_rtol": 1.e-16})
    f_r = Function(V_r).project(mesh_r.coordinates, solver_parameters={"ksp_rtol": 1.e-16})
    A_l = assemble(Constant(1.) * dx(domain=mesh_l))
    A_r = assemble(Constant(1.) * dx(domain=mesh_r))
    A_m = assemble(Constant(1.) * dx(domain=mesh, subdomain_id=middle))
    # Test various combinations.
    x = SpatialCoordinate(mesh)
    f_ = Function(V).assign(f_l, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(left)))
    assert abs(e) / A_l < 1.e-14
    x = SpatialCoordinate(mesh)
    f_ = Function(V).assign(f_r, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(right)))
    assert abs(e) / A_r < 1.e-14
    x = SpatialCoordinate(mesh_l)
    f_ = Function(V_l).assign(f)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(left)))
    assert abs(e) / A_l < 1.e-14
    x = SpatialCoordinate(mesh_r)
    f_ = Function(V_r).assign(f)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(right)))
    assert abs(e) / A_r < 1.e-14
    x = SpatialCoordinate(mesh_l)
    f_ = Function(V_l).assign(f_r, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(middle)))
    assert abs(e) / A_m < 1.e-14
    x = SpatialCoordinate(mesh_r)
    f_ = Function(V_r).assign(f_l, allow_missing_dofs=True)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(middle)))
    assert abs(e) / A_m < 1.e-14


@pytest.mark.parallel(nprocs=2)
def test_submesh_assign_function_subset_3_quads_2_processes():
    left = 111
    right = 222
    middle = 111222
    leftleft = 111111
    rightright = 222222
    mesh = RectangleMesh(
        3, 1, 3., 1., quadrilateral=True, distribution_parameters={"partitioner_type": "simple"},
    )
    dim = mesh.topological_dimension
    x, _ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_l = Function(DG0).interpolate(conditional(x < 2.0, 1, 0))
    f_r = Function(DG0).interpolate(conditional(x > 1.0, 1, 0))
    f_m = Function(DG0).interpolate(conditional(And(x < 2.0, x > 1.0), 1, 0))
    f_ll = Function(DG0).interpolate(conditional(x < 1.0, 1, 0))
    f_rr = Function(DG0).interpolate(conditional(x > 2.0, 1, 0))
    mesh = RelabeledMesh(mesh, [f_l, f_r, f_m, f_ll, f_rr], [left, right, middle, leftleft, rightright])
    mesh_l = Submesh(mesh, dim, left)
    mesh_r = Submesh(mesh, dim, right)
    V = VectorFunctionSpace(mesh, "CG", 3)
    V_l = VectorFunctionSpace(mesh_l, "CG", 3)
    V_r = VectorFunctionSpace(mesh_r, "CG", 3)
    f = Function(V).interpolate(SpatialCoordinate(mesh))
    f_l = Function(V_l).interpolate(SpatialCoordinate(mesh_l))
    f_r = Function(V_r).interpolate(SpatialCoordinate(mesh_r))
    # Test assign on the left two cells.
    # -- mesh_l -> mesh
    subset_indices = np.where(f.dat.data_ro_with_halos[:, 0] < 1.001)
    subset = op2.Subset(f.node_set, subset_indices)
    x = SpatialCoordinate(mesh)
    f_ = Function(V).interpolate(2 * x)
    f_.assign(f_l, subset=subset)
    e = sqrt(assemble(inner(f_ - 2 * x, f_ - 2 * x) * dx(rightright)))
    assert abs(e) < 1.e-14
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(leftleft)))
    assert abs(e) < 1.e-14
    # -- mesh -> mesh_l
    subset_indices = np.where(f_l.dat.data_ro_with_halos[:, 0] < 1.001)
    subset = op2.Subset(f_l.node_set, subset_indices)
    x = SpatialCoordinate(mesh_l)
    f_ = Function(V_l).interpolate(2 * x)
    f_.assign(f, subset=subset)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(leftleft)))
    assert abs(e) < 1.e-14
    # Test assign on the right two cells.
    # -- mesh_r -> mesh
    subset_indices = np.where(f.dat.data_ro_with_halos[:, 0] > 1.999)
    subset = op2.Subset(f.node_set, subset_indices)
    x = SpatialCoordinate(mesh)
    f_ = Function(V).interpolate(2 * x)
    f_.assign(f_r, subset=subset)
    e = sqrt(assemble(inner(f_ - 2 * x, f_ - 2 * x) * dx(leftleft)))
    assert abs(e) < 1.e-14
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(rightright)))
    assert abs(e) < 1.e-14
    # -- mesh -> mesh_r
    subset_indices = np.where(f_r.dat.data_ro_with_halos[:, 0] > 1.999)
    subset = op2.Subset(f_r.node_set, subset_indices)
    x = SpatialCoordinate(mesh_r)
    f_ = Function(V_r).interpolate(2 * x)
    f_.assign(f, subset=subset)
    e = sqrt(assemble(inner(f_ - x, f_ - x) * dx(rightright)))
    assert abs(e) < 1.e-14


@pytest.mark.parallel(nprocs=2)
def test_submesh_assign_cofunction_3_quads_2_processes():
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
    dim = mesh.topological_dimension
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
    v = TestFunction(V)
    v_l = TestFunction(V_l)
    v_r = TestFunction(V_r)
    coords = Function(V).interpolate(SpatialCoordinate(mesh))
    coords_l = Function(V_l).interpolate(SpatialCoordinate(mesh_l))
    coords_r = Function(V_r).interpolate(SpatialCoordinate(mesh_r))
    cof = assemble(inner(SpatialCoordinate(mesh), v) * dx)
    cof_l = assemble(inner(SpatialCoordinate(mesh_l), v_l) * dx)
    cof_r = assemble(inner(SpatialCoordinate(mesh_r), v_r) * dx)
    # Test assign on the left two cells.
    # -- mesh_l -> mesh
    cof_ = Cofunction(V.dual()).assign(cof_l, allow_missing_dofs=True)
    subset_indices = np.where(coords.dat.data_ro_with_halos[:, 0] < 1.001)
    assert np.allclose(cof_.dat.data_ro_with_halos[subset_indices], cof.dat.data_ro_with_halos[subset_indices])
    # -- mesh -> mesh_l
    cof_ = Cofunction(V_l.dual()).assign(cof)
    subset_indices = np.where(coords_l.dat.data_ro_with_halos[:, 0] < 1.001)
    assert np.allclose(cof_.dat.data_ro_with_halos[subset_indices], cof_l.dat.data_ro_with_halos[subset_indices])
    # Test assign on the right two cells.
    # -- mesh_r -> mesh
    cof_ = Cofunction(V.dual()).assign(cof_r, allow_missing_dofs=True)
    subset_indices = np.where(coords.dat.data_ro_with_halos[:, 0] > 1.999)
    assert np.allclose(cof_.dat.data_ro_with_halos[subset_indices], cof.dat.data_ro_with_halos[subset_indices])
    # -- mesh -> mesh_r
    cof_ = Cofunction(V_r.dual()).assign(cof)
    subset_indices = np.where(coords_r.dat.data_ro_with_halos[:, 0] > 1.999)
    assert np.allclose(cof_.dat.data_ro_with_halos[subset_indices], cof_r.dat.data_ro_with_halos[subset_indices])


@pytest.mark.parametrize("mesh_type,family,degree", [
    ("simplex", "CG", 3),
    ("simplex", "RT", 2),
    ("simplex", "N1curl", 1),
    ("quadrilateral", "CG", 2),
    ("quadrilateral", "RTCF", 1),
])
@pytest.mark.parallel(nprocs=[1, 3])
def test_submesh_assign_function_redistributed(mesh_type, family, degree):
    # A redistributed submesh holds the same mesh as its parent. Assigning
    # between the two must therefore be exact for every element. That includes
    # the elements whose DoFs depend on the orientation of their entity.
    mesh = UnitSquareMesh(4, 4, quadrilateral=(mesh_type == "quadrilateral"))
    submesh = Submesh(mesh, redistribute=True)
    assert submesh.topology.submesh_parent is mesh.topology
    assert submesh.topology.is_redistributed

    V = FunctionSpace(mesh, family, degree)
    V_sub = FunctionSpace(submesh, family, degree)
    x, y = SpatialCoordinate(mesh)
    x_sub, y_sub = SpatialCoordinate(submesh)
    if V.value_size == 1:
        expr, expr_sub = sin(x) * cos(3 * y), sin(x_sub) * cos(3 * y_sub)
    else:
        expr = as_vector([sin(x), cos(3 * y)])
        expr_sub = as_vector([sin(x_sub), cos(3 * y_sub)])
    f = Function(V).interpolate(expr)
    # -- mesh -> submesh
    f_sub = Function(V_sub).assign(f)
    assert np.allclose(norm(f_sub - Function(V_sub).interpolate(expr_sub)), 0)
    # -- submesh -> mesh
    assert np.allclose(norm(Function(V).assign(f_sub) - f), 0)
    # -- the dual assignment preserves the action
    cof_sub = assemble(inner(expr_sub, TestFunction(V_sub)) * dx)
    cof = Cofunction(V.dual()).assign(cof_sub)
    assert np.isclose(assemble(action(cof, f)), assemble(action(cof_sub, f_sub)))


@pytest.mark.parametrize("family,degree", [("CG", 2), ("RT", 1)])
@pytest.mark.parallel(nprocs=[1, 3])
def test_submesh_assign_function_redistributed_subdomain(family, degree):
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    mesh.mark_entities(Function(DG0).interpolate(conditional(x < 0.5, 1, 0)), 111)
    submesh = Submesh(mesh, subdomain_id=111, redistribute=True)

    V = FunctionSpace(mesh, family, degree)
    V_sub = FunctionSpace(submesh, family, degree)
    x_sub, y_sub = SpatialCoordinate(submesh)
    if V.value_size == 1:
        expr, expr_sub = sin(x) * cos(3 * y), sin(x_sub) * cos(3 * y_sub)
    else:
        expr = as_vector([sin(x), cos(3 * y)])
        expr_sub = as_vector([sin(x_sub), cos(3 * y_sub)])
    f = Function(V).interpolate(expr)
    # -- mesh -> submesh
    f_sub = Function(V_sub).assign(f)
    assert np.allclose(norm(f_sub - Function(V_sub).interpolate(expr_sub)), 0)
    # -- submesh -> mesh: the parent has nodes outside the subdomain
    with pytest.raises(ValueError):
        Function(V).assign(f_sub)
    g = Function(V).interpolate(expr)
    g.assign(f_sub, allow_missing_dofs=True)
    assert np.allclose(norm(g - f), 0)


@pytest.mark.parallel(nprocs=[1, 3])
def test_submesh_assign_cell_subset_redistributed():
    # A cell subset of the parent mesh assigned from a redistributed submesh
    # that does not share the parent's parallel distribution.
    sentinel = -1.0
    mesh = UnitSquareMesh(6, 6)
    x, y = SpatialCoordinate(mesh)
    marker = Function(FunctionSpace(mesh, "DG", 0)).interpolate(conditional(x < 0.5, 1, 0))
    mesh.mark_entities(marker, 7)

    V = FunctionSpace(mesh, "CG", 2)
    source = Function(V).interpolate(sin(3 * x))
    submesh = Submesh(mesh, subdomain_id=7, redistribute=True)
    x_sub, _ = SpatialCoordinate(submesh)
    f_sub = Function(FunctionSpace(submesh, "CG", 2)).interpolate(sin(3 * x_sub))
    composed = Function(V).assign(sentinel)
    composed.assign(f_sub, subset=mesh.cell_subset(7), allow_missing_dofs=True)
    written = composed.dat.data_ro != sentinel
    assert np.allclose(composed.dat.data_ro[written], source.dat.data_ro[written])
    assert mesh.comm.allreduce(int(written[:V.dof_dset.size].sum())) == 91


@pytest.mark.parametrize("family,degree", [("CG", 3), ("RT", 2)])
@pytest.mark.parametrize("redistribute", [False, True])
@pytest.mark.parallel(nprocs=[1, 3])
def test_submesh_assign_composed_restrictions(family, degree, redistribute):
    # Compose all three restrictions of the node layout at once. Each of them
    # applies to one side of the assignment only. The source is the whole of
    # an element on the whole of the parent mesh. The target restricts the mesh
    # to a proper subdomain, optionally redistributing it. It then restricts to
    # the boundary, and the element to the facets of the reference cell.
    sentinel = -12345.0
    mesh = UnitSquareMesh(6, 6)
    x, y = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    mesh.mark_entities(Function(DG0).interpolate(conditional(x < 0.5, 1, 0)), 111)
    submesh = Submesh(mesh, subdomain_id=111, redistribute=redistribute)

    elem = finat.ufl.FiniteElement(family, mesh.ufl_cell(), degree)
    V = FunctionSpace(mesh, elem)
    V_sub = RestrictedFunctionSpace(
        FunctionSpace(submesh, finat.ufl.RestrictedElement(elem, restriction_domain="facet")),
        boundary_set={1},
    )

    f = Function(V)
    f.dat.data_wo[:] = 1.0 + np.arange(f.dat.data_ro.size)

    f_sub = Function(V_sub).assign(f)
    g = Function(V).assign(sentinel)
    g.assign(f_sub, allow_missing_dofs=True)

    covered = g.dat.data_ro != sentinel
    assert mesh.comm.allreduce(int(covered.sum())) > 0
    assert np.allclose(g.dat.data_ro[covered], f.dat.data_ro[covered])


@pytest.mark.parametrize("cell,family,degree", [
    ("triangle", "CG", 3),
    ("triangle", "RT", 2),
    ("quadrilateral", "Q", 3),
])
@pytest.mark.parallel(nprocs=[1, 3])
def test_assign_restricted_element_same_mesh(cell, family, degree):
    # An element and its restriction lay their nodes out differently on the
    # very same mesh; the two are related by their Sections alone.
    sentinel = -12345.0
    mesh = UnitSquareMesh(6, 6, quadrilateral=(cell == "quadrilateral"))
    elem = finat.ufl.FiniteElement(family, mesh.ufl_cell(), degree)
    V = FunctionSpace(mesh, elem)
    V_facet = FunctionSpace(mesh, finat.ufl.RestrictedElement(elem, restriction_domain="facet"))

    x, y = SpatialCoordinate(mesh)
    expr = sin(3 * x) + 2 * cos(5 * y)
    if V.value_shape:
        expr = as_vector([sin(3 * x), cos(5 * y)])
    f = Function(V).interpolate(expr)

    # -- the restriction keeps the facet nodes of the full element. Compare
    # against interpolation, which numbers the nodes of the restricted space
    # without reference to the parent. A permutation of the nodes within an
    # entity therefore cannot go unnoticed.
    f_facet = Function(V_facet).assign(f)
    assert np.allclose(f_facet.dat.data_ro, Function(V_facet).interpolate(expr).dat.data_ro)

    # -- and back: the full element has interior nodes with no counterpart
    with pytest.raises(ValueError):
        Function(V).assign(f_facet)
    g = Function(V).assign(sentinel)
    g.assign(f_facet, allow_missing_dofs=True)

    covered = g.dat.data_ro != sentinel
    assert mesh.comm.allreduce(int(covered[:V.dof_dset.size].sum())) == V_facet.dim()
    assert np.allclose(g.dat.data_ro[covered], f.dat.data_ro[covered])


@pytest.mark.parametrize("shape", ["vector", "symmetric"])
@pytest.mark.parallel(nprocs=[1, 3])
def test_assign_restricted_element_blocked(shape):
    # A vector or tensor element holds a block of the nodes of a single
    # sub-element, and UFL pushes the restriction inside the block.
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    elem = finat.ufl.FiniteElement("CG", mesh.ufl_cell(), 3)
    restricted = finat.ufl.RestrictedElement(elem, restriction_domain="facet")
    if shape == "vector":
        block, expr = finat.ufl.VectorElement, as_vector([sin(3 * x), cos(5 * y)])
    else:
        block = partial(finat.ufl.TensorElement, symmetry=True)
        expr = as_tensor([[sin(3 * x), cos(5 * y)], [cos(5 * y), x - y]])

    V_facet = FunctionSpace(mesh, block(restricted))
    f = Function(FunctionSpace(mesh, block(elem))).interpolate(expr)
    assert np.allclose(Function(V_facet).assign(f).dat.data_ro,
                       Function(V_facet).interpolate(expr).dat.data_ro)


@pytest.mark.parallel(nprocs=1)
def test_assign_incompatible_elements():
    mesh = UnitSquareMesh(2, 2)
    elem = finat.ufl.FiniteElement("CG", mesh.ufl_cell(), 3)

    def facet_space(mesh, element):
        return FunctionSpace(mesh, finat.ufl.RestrictedElement(element, restriction_domain="facet"))

    f = Function(FunctionSpace(mesh, "CG", 1))
    with pytest.raises(ValueError):
        f.assign(Function(FunctionSpace(mesh, "RT", 1)))
    # CG1 carries the nodes CG2 puts on the vertices, and drops the rest. They
    # are not the nodes of a common element, so this is not an assignment.
    with pytest.raises(ValueError):
        f.assign(Function(FunctionSpace(mesh, "CG", 2)))

    # Restricting to the facets of an interval leaves one node per vertex,
    # whatever the degree. The node counts alone therefore cannot tell the two
    # elements apart. Only the element they restrict can.
    interval = UnitIntervalMesh(4)
    cells = [finat.ufl.FiniteElement("CG", interval.ufl_cell(), d) for d in (2, 3)]
    with pytest.raises(ValueError):
        Function(facet_space(interval, cells[1])).assign(Function(facet_space(interval, cells[0])))

    # Blocks of different shape hold different numbers of nodes.
    vector = Function(FunctionSpace(mesh, finat.ufl.VectorElement(elem, dim=2)))
    for other in (finat.ufl.VectorElement(elem, dim=3), finat.ufl.TensorElement(elem), elem):
        with pytest.raises(ValueError):
            vector.assign(Function(FunctionSpace(mesh, other)))

    # A base mesh point of an extruded mesh carries a whole column of
    # entities, of which the restriction drops only some.
    extruded = ExtrudedMesh(UnitSquareMesh(2, 2, quadrilateral=True), 2)
    q = finat.ufl.FiniteElement("Q", extruded.ufl_cell(), 3)
    with pytest.raises(NotImplementedError):
        Function(facet_space(extruded, q)).assign(Function(FunctionSpace(extruded, q)))


@pytest.mark.parallel(nprocs=[1, 3])
def test_assign_multiple_source_spaces():
    # An interior and a facet restriction of one element are each compatible
    # with the parent, but not with one another. Each term of the sum is
    # therefore related to the assignee by its own section SF. Both are moved
    # into that layout before they are added.
    mesh = UnitSquareMesh(6, 6)
    elem = finat.ufl.FiniteElement("CG", mesh.ufl_cell(), 3)
    V = FunctionSpace(mesh, elem)
    V_interior = FunctionSpace(mesh, finat.ufl.RestrictedElement(elem, restriction_domain="interior"))
    V_facet = FunctionSpace(mesh, finat.ufl.RestrictedElement(elem, restriction_domain="facet"))

    x, y = SpatialCoordinate(mesh)
    expr = sin(3 * x) + 2 * cos(5 * y)
    f = Function(V).interpolate(expr)
    f_interior = Function(V_interior).assign(f)
    f_facet = Function(V_facet).assign(f)

    g = Function(V)
    g.assign(f_interior + f_facet)
    assert np.allclose(g.dat.data_ro, f.dat.data_ro)
