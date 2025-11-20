import os
import pytest
import numpy as np
from firedrake import utils
from firedrake import *
from firedrake.cython import dmcommon
from petsc4py import PETSc


cwd = os.path.abspath(os.path.dirname(__file__))


def get_sparsity(mat, *nest_indices):
    subpetscmat = mat.petscmat.getNestSubMatrix(*nest_indices)
    row_ptrs, _ = subpetscmat.getRowIJ()
    row_sizes = np.full(len(row_ptrs)-1, -1, dtype=int)
    for row_index, (row_start, row_end) in enumerate(utils.pairwise(row_ptrs)):
        row_sizes[row_index] = row_end - row_start
    return row_sizes


def get_values(mat, *nest_indices):
    subpetscmat = mat.petscmat.getNestSubMatrix(*nest_indices)
    return subpetscmat[:, :]


def test_submesh_assemble_cell_cell_integral_cell():
    dim = 2
    mesh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(conditional(x > 1., 1, 0))
    mesh.mark_entities(indicator_function, 999)
    subm = Submesh(mesh, dim, 999)
    V0 = FunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(subm, "CG", 1)
    V = V0 * V1
    u = TrialFunction(V)
    v = TestFunction(V)
    u0, u1 = split(u)
    v0, v1 = split(v)
    dx0 = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", subm),))
    dx1 = Measure("dx", domain=subm, intersect_measures=(Measure("dx", mesh),))
    a = inner(u1, v0) * dx0(999) + inner(u0, v1) * dx1
    A = assemble(a, mat_type="nest")

    assert np.allclose(get_sparsity(A, 0, 0), [1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(get_sparsity(A, 0, 1), [4, 4, 4, 4, 0, 0])
    assert np.allclose(get_sparsity(A, 1, 0), [4, 4, 4, 4])
    assert np.allclose(get_sparsity(A, 1, 1), [1, 1, 1, 1])  # bc nodes

    M10 = np.array([[1./9. , 1./18., 1./36., 1./18., 0., 0.],   # noqa: E203
                    [1./18., 1./9. , 1./18., 1./36., 0., 0.],   # noqa: E203
                    [1./36., 1./18., 1./9. , 1./18., 0., 0.],   # noqa: E203
                    [1./18., 1./36., 1./18., 1./9. , 0., 0.]])  # noqa: E203
    assert np.allclose(get_values(A, 0, 1), np.transpose(M10))
    assert np.allclose(get_values(A, 1, 0), M10)


def test_submesh_assemble_cell_cell_integral_facet():
    dim = 2
    mesh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(conditional(x > 1., 1, 0))
    mesh.mark_entities(indicator_function, 999)
    subm = Submesh(mesh, dim, 999)
    V0 = FunctionSpace(mesh, "DQ", 1, variant="equispaced")
    V1 = FunctionSpace(subm, "DQ", 1, variant="equispaced")
    V = V0 * V1
    u = TrialFunction(V)
    v = TestFunction(V)
    u0, u1 = split(u)
    v0, v1 = split(v)
    dS0 = Measure("dS", domain=mesh, intersect_measures=(Measure("ds", subm),))
    ds1 = Measure("ds", domain=subm, intersect_measures=(Measure("dS", mesh),))
    a = inner(u1, v0('+')) * dS0 + inner(u0('+'), v1) * ds1(5)
    A = assemble(a, mat_type="nest")
    assert np.allclose(get_sparsity(A, 0, 0), [1, 1, 1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(get_sparsity(A, 0, 1), [4, 4, 4, 4, 4, 4, 4, 4])
    assert np.allclose(get_sparsity(A, 1, 0), [8, 8, 8, 8])
    assert np.allclose(get_sparsity(A, 1, 1), [1, 1, 1, 1])  # bc nodes

    M10 = np.array([[0., 0., 0., 0., 0., 0., 1. / 3., 1. / 6.],
                    [0., 0., 0., 0., 0., 0., 1. / 6., 1. / 3.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.]])
    assert np.allclose(get_values(A, 0, 1), np.transpose(M10))
    assert np.allclose(get_values(A, 1, 0), M10)

    b = inner(u1, v0('+')) * ds1(5) + inner(u0('+'), v1) * dS0
    B = assemble(b, mat_type="nest")
    assert np.allclose(get_sparsity(B, 0, 0), [1, 1, 1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(get_sparsity(B, 0, 1), [4, 4, 4, 4, 4, 4, 4, 4])
    assert np.allclose(get_sparsity(B, 1, 0), [8, 8, 8, 8])
    assert np.allclose(get_sparsity(B, 1, 1), [1, 1, 1, 1])  # bc nodes
    assert np.allclose(get_values(B, 0, 1), get_values(A, 0, 1))
    assert np.allclose(get_values(B, 1, 0), get_values(A, 1, 0))


def test_submesh_assemble_cell_cell_cell_cell_integral_various():
    #        +-------+-------+-------+-------+
    #        |       |       |       |       |
    #        |       |      555      |       |    mesh
    #        |       |       |       |       |
    #        +-------+-------+-------+-------+
    #        +-------+-------+
    #        |       |       |
    #        |       |      555                   mesh_l
    #        |       |       |
    #        +-------+-------+
    #                        +-------+-------+
    #                        |       |       |
    #                       555      |       |    mesh_r
    #                        |       |       |
    #                        +-------+-------+
    #                        +-------+
    #                        |       |
    #                       555      |            mesh_rl
    #                        |       |
    #                        +-------+
    dim = 2
    mesh = RectangleMesh(4, 1, 4., 1., quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    label_int = 555
    label_l = 81100
    label_r = 80011
    label_rl = 80010
    HDivTrace0 = FunctionSpace(mesh, "HDiv Trace", 0)
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_int = Function(HDivTrace0).interpolate(conditional(And(x > 1.9, x < 2.1), 1, 0))
    f_l = Function(DG0).interpolate(conditional(x < 2., 1, 0))
    f_r = Function(DG0).interpolate(conditional(x > 2., 1, 0))
    f_rl = Function(DG0).interpolate(conditional(And(x > 2., x < 3.), 1, 0))
    mesh = RelabeledMesh(mesh, [f_int, f_l, f_r, f_rl], [label_int, label_l, label_r, label_rl])
    x, y = SpatialCoordinate(mesh)
    mesh_l = Submesh(mesh, dim, label_l)
    mesh_r = Submesh(mesh, dim, label_r)
    mesh_rl = Submesh(mesh_r, dim, label_rl)
    dS = Measure(
        "dS", domain=mesh,
        intersect_measures=(
            Measure("ds", mesh_l),
            Measure("ds", mesh_r),
            Measure("ds", mesh_rl),
        )
    )
    ds_l = Measure(
        "ds", domain=mesh_l,
        intersect_measures=(
            Measure("dS", mesh),
            Measure("ds", mesh_r),
            Measure("ds", mesh_rl),
        )
    )
    ds_r = Measure(
        "ds", domain=mesh_r,
        intersect_measures=(
            Measure("dS", mesh),
            Measure("ds", mesh_l),
            Measure("ds", mesh_rl),
        )
    )
    ds_rl = Measure(
        "ds", domain=mesh_rl,
        intersect_measures=(
            Measure("dS", mesh),
            Measure("ds", mesh_l),
            Measure("ds", mesh_r),
        )
    )
    n_l = FacetNormal(mesh_l)
    n_rl = FacetNormal(mesh_rl)
    assert assemble(dot(n_rl + n_l, n_rl + n_l) * ds_rl(label_int)) < 1.e-32
    assert assemble(dot(n_rl + n_l, n_rl + n_l) * ds_r(label_int)) < 1.e-32
    assert assemble(dot(n_rl + n_l, n_rl + n_l) * ds_l(label_int)) < 1.e-32
    assert assemble(dot(n_rl + n_l, n_rl + n_l) * dS(label_int)) < 1.e-32
    V_l = FunctionSpace(mesh_l, "DQ", 1, variant='equispaced')
    V_rl = FunctionSpace(mesh_rl, "DQ", 1, variant='equispaced')
    V = V_l * V_rl
    u_l, u_rl = TrialFunctions(V)
    v_l, v_rl = TestFunctions(V)
    a = inner(u_rl, v_l) * ds_l(label_int) + inner(u_l, v_rl) * ds_rl(label_int)
    A = assemble(a, mat_type="nest")
    assert np.allclose(get_sparsity(A, 0, 0), [1, 1, 1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(get_sparsity(A, 0, 1), [4, 4, 4, 4, 0, 0, 0, 0])
    assert np.allclose(get_sparsity(A, 1, 0), [4, 4, 4, 4])
    assert np.allclose(get_sparsity(A, 1, 1), [1, 1, 1, 1])  # bc nodes
    M10 = np.array([[0., 0., 1. / 3., 1. / 6., 0., 0., 0., 0.],
                    [0., 0., 1. / 6., 1. / 3., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.]])
    assert np.allclose(get_values(A, 0, 1), np.transpose(M10))
    assert np.allclose(get_values(A, 1, 0), M10)
    b = inner(u_rl, v_l) * dS(label_int) + inner(u_l, v_rl) * dS(label_int)
    B = assemble(b, mat_type="nest")
    assert np.allclose(get_sparsity(B, 0, 0), [1, 1, 1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(get_sparsity(B, 0, 1), [4, 4, 4, 4, 0, 0, 0, 0])
    assert np.allclose(get_sparsity(B, 1, 0), [4, 4, 4, 4])
    assert np.allclose(get_sparsity(B, 1, 1), [1, 1, 1, 1])  # bc nodes
    assert np.allclose(get_values(B, 0, 1), get_values(A, 0, 1))
    assert np.allclose(get_values(B, 1, 0), get_values(A, 1, 0))


def test_submesh_assemble_cell_cell_cell_cell_integral_avg():
    #        +-------+-------+-------+-------+
    #        |       |       |       |       |
    #        |       |      555      |       |    mesh
    #        |       |       |       |       |
    #        +-------+-------+-------+-------+
    #        +-------+-------+-------+
    #        |       |       |       |
    #        |       |      555      |            mesh_l
    #        |       |       |       |
    #        +-------+-------+-------+
    #                        +-------+-------+
    #                        |       |       |
    #                       555      |       |    mesh_r
    #                        |       |       |
    #                        +-------+-------+
    #                        +-------+
    #                        |       |
    #                       555      |            mesh_rl
    #                        |       |
    #                        +-------+
    dim = 2
    mesh = RectangleMesh(4, 1, 4., 1., quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    label_int = 555
    label_l = 81110
    label_r = 80011
    label_rl = 80010
    HDivTrace0 = FunctionSpace(mesh, "HDiv Trace", 0)
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_int = Function(HDivTrace0).interpolate(conditional(And(x > 1.9, x < 2.1), 1, 0))
    f_l = Function(DG0).interpolate(conditional(x < 3., 1, 0))
    f_r = Function(DG0).interpolate(conditional(x > 2., 1, 0))
    f_rl = Function(DG0).interpolate(conditional(And(x > 2., x < 3.), 1, 0))
    mesh = RelabeledMesh(mesh, [f_int, f_l, f_r, f_rl], [label_int, label_l, label_r, label_rl])
    x, y = SpatialCoordinate(mesh)
    mesh_l = Submesh(mesh, dim, label_l)
    x_l, y_l = SpatialCoordinate(mesh_l)
    mesh_r = Submesh(mesh, dim, label_r)
    x_r, y_r = SpatialCoordinate(mesh_r)
    mesh_rl = Submesh(mesh_r, dim, label_rl)
    x_rl, y_rl = SpatialCoordinate(mesh_rl)
    dx = Measure(
        "dx", domain=mesh,
        intersect_measures=(
            Measure("dx", mesh_l),
            Measure("dx", mesh_r),
            Measure("dx", mesh_rl),
        )
    )
    dx_l = Measure(
        "dx", domain=mesh_l,
        intersect_measures=(
            Measure("dx", mesh),
            Measure("dx", mesh_r),
            Measure("dx", mesh_rl),
        )
    )
    dx_rl = Measure(
        "dx", domain=mesh_rl,
        intersect_measures=(
            Measure("dx", mesh),
            Measure("dx", mesh_l),
            Measure("dx", mesh_r),
        )
    )
    dS = Measure(
        "dS", domain=mesh,
        intersect_measures=(
            Measure("dS", mesh_l),
            Measure("ds", mesh_r),
            Measure("ds", mesh_rl),
        )
    )
    dS_l = Measure(
        "dS", domain=mesh_l,
        intersect_measures=(
            Measure("dS", mesh),
            Measure("ds", mesh_r),
            Measure("ds", mesh_rl),
        )
    )
    ds_rl = Measure(
        "ds", domain=mesh_rl,
        intersect_measures=(
            Measure("dS", mesh),
            Measure("dS", mesh_l),
            Measure("ds", mesh_r),
        )
    )
    assert abs(assemble(cell_avg(x) * dx(label_rl)) - 2.5) < 5.e-16
    assert abs(assemble(cell_avg(x) * dx_rl) - 2.5) < 5.e-16
    assert abs(assemble(cell_avg(x_rl) * dx(label_rl)) - 2.5) < 5.e-16
    assert abs(assemble(cell_avg(x_rl) * dx_l(label_rl)) - 2.5) < 5.e-16
    assert abs(assemble(cell_avg(x_l) * dx_rl) - 2.5) < 5.e-16
    assert abs(assemble(facet_avg(y * y) * dS(label_int)) - 1. / 3.) < 5.e-16
    assert abs(assemble(facet_avg(y('+') * y('-')) * ds_rl(label_int)) - 1. / 3.) < 5.e-16
    assert abs(assemble(facet_avg(y_rl * y_rl) * dS(label_int)) - 1. / 3.) < 5.e-16
    assert abs(assemble(facet_avg(y_rl * y_rl) * dS_l(label_int)) - 1. / 3.) < 5.e-16
    assert abs(assemble(facet_avg(y_l('+') * y_l('-')) * ds_rl(label_int)) - 1. / 3.) < 5.e-16


def test_submesh_assemble_cell_cell_equation_bc():
    dim = 2
    mesh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    label_int = 555
    label_l = 810
    label_r = 801
    HDivTrace0 = FunctionSpace(mesh, "HDiv Trace", 0)
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    f_int = Function(HDivTrace0).interpolate(conditional(And(x > 0.9, x < 1.1), 1, 0))
    f_l = Function(DQ0).interpolate(conditional(x < 1., 1, 0))
    f_r = Function(DQ0).interpolate(conditional(x > 1., 1, 0))
    mesh = RelabeledMesh(mesh, [f_int, f_l, f_r], [label_int, label_l, label_r])
    mesh_l = Submesh(mesh, dim, label_l)
    mesh_r = Submesh(mesh, dim, label_r)
    V_l = FunctionSpace(mesh_l, "CG", 1)
    V_r = FunctionSpace(mesh_r, "CG", 1)
    V = V_l * V_r
    u = TrialFunction(V)
    v = TestFunction(V)
    u_l, u_r = split(u)
    v_l, v_r = split(v)
    dx_l = Measure("dx", domain=mesh_l)
    ds_l = Measure("ds", domain=mesh_l, intersect_measures=(Measure("ds", mesh_r),))
    a = inner(u_l, v_l) * dx_l
    a_int = inner(u_l - u_r, v_l) * ds_l(label_int)
    L_int = inner(Constant(0), v_l) * ds_l(label_int)
    sol = Function(V)
    bc = EquationBC(a_int == L_int, sol, label_int, V=V.sub(0))
    A = assemble(a, bcs=bc.extract_form('J'), mat_type="nest")
    assert np.allclose(Function(V_l).interpolate(SpatialCoordinate(mesh_l)[0]).dat.data, [0., 0., 1., 1.])
    assert np.allclose(Function(V_l).interpolate(SpatialCoordinate(mesh_l)[1]).dat.data, [0., 1., 1., 0.])
    assert np.allclose(Function(V_r).interpolate(SpatialCoordinate(mesh_r)[0]).dat.data, [1., 1., 2., 2.])
    assert np.allclose(Function(V_r).interpolate(SpatialCoordinate(mesh_r)[1]).dat.data, [0., 1., 1., 0.])
    assert np.allclose(get_sparsity(A, 0, 0), [4, 4, 4, 4])
    assert np.allclose(get_sparsity(A, 0, 1), [4, 4, 4, 4])
    assert np.allclose(get_sparsity(A, 1, 0), [0, 0, 0, 0])
    assert np.allclose(get_sparsity(A, 1, 1), [1, 1, 1, 1])  # bc nodes
    M00 = np.array([[1. / 9. , 1. / 18., 1. / 36., 1. / 18.],  # noqa: E203
                    [1. / 18., 1. / 9. , 1. / 18., 1. / 36.],  # noqa: E203
                    [0., 0., 1. / 3., 1. / 6.],
                    [0., 0., 1. / 6., 1. / 3.]])
    M01 = np.array([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [- 1. / 6., - 1. / 3., 0., 0.],
                    [- 1. / 3., - 1. / 6., 0., 0.]])
    assert np.allclose(get_values(A, 0, 0), M00)
    assert np.allclose(get_values(A, 0, 1), M01)


def test_submesh_assemble_cell_facet_integral_various():
    #  CG1 DoF numbers (nprocs = 1):
    #
    #  5-------1-------2
    #  |       |       |
    #  |       |       |  mesh
    #  |       |       |
    #  4-------0-------3
    #
    #          0
    #          |
    #          |          subm
    #          |
    #          1
    #
    distribution_parameters = {
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
    }
    subdomain_id = 777
    mesh = RectangleMesh(2, 1, 2., 1., quadrilateral=True, distribution_parameters=distribution_parameters)
    x, y = SpatialCoordinate(mesh)
    V1 = FunctionSpace(mesh, "HDiv Trace", 0)
    f1 = Function(V1).interpolate(conditional(And(x > 0.9, x < 1.1), 1., 0.))
    mesh = RelabeledMesh(mesh, [f1], [subdomain_id])
    x, y = SpatialCoordinate(mesh)
    subm = Submesh(mesh, mesh.topological_dimension - 1, subdomain_id)
    subx, suby = SpatialCoordinate(subm)
    V0 = FunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(subm, "CG", 1)
    V = V0 * V1
    u = TrialFunction(V)
    v = TestFunction(V)
    u0, u1 = split(u)
    v0, v1 = split(v)
    coordV0 = VectorFunctionSpace(mesh, "CG", 1)
    coordV1 = VectorFunctionSpace(subm, "CG", 1)
    coordV = coordV0 * coordV1
    coords = Function(coordV)
    coords.sub(0).assign(mesh.coordinates)
    coords.sub(1).assign(subm.coordinates)
    coords0, coords1 = split(coords)
    M10 = np.array(
        [
            [1. / 6., 1. / 3., 0., 0., 0., 0.],
            [1. / 3., 1. / 6., 0., 0., 0., 0.],
        ]
    )
    M10w = np.array(
        [
            [1. / 12., 1. / 4., 0., 0., 0., 0.],
            [1. / 12., 1. / 12., 0., 0., 0., 0.],
        ]
    )
    M10ww = np.array(
        [
            [1. / 20., 1. / 5., 0., 0., 0., 0.],
            [1. / 30., 1. / 20., 0., 0., 0., 0.],
        ]
    )
    # Use subm as primal integration domain.
    measure = Measure(
        "dx", subm,
        intersect_measures=(
            Measure("dS", mesh),
        ),
    )
    a = inner(u0('-'), v1) * measure
    A = assemble(a, mat_type="nest")
    assert np.allclose(get_values(A, 1, 0), M10)
    a = inner(u1, v0('+')) * measure
    A = assemble(a, mat_type="nest")
    assert np.allclose(get_values(A, 0, 1), np.transpose(M10))
    a = y * inner(u0('-'), v1) * measure
    A = assemble(a, mat_type="nest")
    assert np.allclose(get_values(A, 1, 0), M10w)
    a = y * suby * inner(u0('-'), v1) * measure
    A = assemble(a, mat_type="nest")
    assert np.allclose(get_values(A, 1, 0), M10ww)
    a = coords0[1] * inner(u0('-'), v1) * measure
    A = assemble(a, mat_type="nest")
    assert np.allclose(get_values(A, 1, 0), M10w)
    a = coords0[1] * coords1[1] * inner(u0('-'), v1) * measure
    A = assemble(a, mat_type="nest")
    assert np.allclose(get_values(A, 1, 0), M10ww)
    # Use mesh as primal integration domain.
    measure = Measure(
        "dS", mesh,
        intersect_measures=(
            Measure("dx", subm),
        ),
    )
    a = inner(u0('+'), v1) * measure(subdomain_id)
    A = assemble(a, mat_type="nest")
    assert np.allclose(get_values(A, 1, 0), M10)
    a = inner(u1, v0('-')) * measure(subdomain_id)
    A = assemble(a, mat_type="nest")
    assert np.allclose(get_values(A, 0, 1), np.transpose(M10))


@pytest.mark.parallel([1, 2, 3])
def test_submesh_assemble_quad_triangle_base():
    dim = 2
    label_ext = 1
    label_interf = 2
    mesh = Mesh(os.path.join(cwd, "..", "meshes", "mixed_cell_unit_square.msh"))
    mesh.topology_dm.markBoundaryFaces(dmcommon.FACE_SETS_LABEL, label_ext)
    mesh_t = Submesh(mesh, dim, PETSc.DM.PolytopeType.TRIANGLE, label_name="celltype", name="mesh_tri")
    x_t, y_t = SpatialCoordinate(mesh_t)
    n_t = FacetNormal(mesh_t)
    mesh_q = Submesh(mesh, dim, PETSc.DM.PolytopeType.QUADRILATERAL, label_name="celltype", name="mesh_quad")
    x_q, y_q = SpatialCoordinate(mesh_q)
    n_q = FacetNormal(mesh_q)
    # pgfplot(f, "mesh_tri.dat", degree=2)
    dx_t = Measure("dx", mesh_t)
    dx_q = Measure("dx", mesh_q)
    ds_t = Measure("ds", mesh_t, intersect_measures=(Measure("ds", mesh_q),))
    ds_q = Measure("ds", mesh_q, intersect_measures=(Measure("ds", mesh_t),))
    A_t = assemble(Constant(1) * dx_t)
    A_q = assemble(Constant(1) * dx_q)
    assert abs(A_t + A_q - 1.0) < 1.e-13
    HDiv_t = FunctionSpace(mesh_t, "BDM", 3)
    HDiv_q = FunctionSpace(mesh_q, "RTCF", 3)
    hdiv_t = Function(HDiv_t).interpolate(as_vector([x_t**2, y_t**2]))
    hdiv_q = Function(HDiv_q).project(as_vector([x_q**2, y_q**2]), solver_parameters={"ksp_rtol": 1.e-13})
    v_t = assemble(dot(hdiv_q, as_vector([x_q, y_q])) * ds_t(label_interf))
    v_q = assemble(dot(hdiv_t, as_vector([x_t, y_t])) * ds_q(label_interf))
    assert abs(v_q - v_t) < 1.e-13
    v_t = assemble(dot(hdiv_q, as_vector([x_t, y_t])) * ds_t(label_interf))
    v_q = assemble(dot(hdiv_t, as_vector([x_q, y_q])) * ds_q(label_interf))
    assert abs(v_q - v_t) < 1.e-13
    v_t = assemble(dot(hdiv_q, as_vector([x_q, y_t])) * ds_t(label_interf))
    v_q = assemble(dot(hdiv_t, as_vector([x_t, y_q])) * ds_q(label_interf))
    assert abs(v_q - v_t) < 1.e-13
    v = assemble(inner(n_t, as_vector([888., 999.])) * ds_t(label_interf))
    assert abs(v) < 1.e-13
    v = assemble(inner(n_q, as_vector([888., 999.])) * ds_q(label_interf))
    assert abs(v) < 1.e-13
    v = assemble(inner(n_q, as_vector([888., 999.])) * ds_t(label_interf))
    assert abs(v) < 1.e-13
    v = assemble(inner(n_t, as_vector([888., 999.])) * ds_q(label_interf))
    assert abs(v) < 1.e-13
    v = assemble(dot(n_q + n_t, n_q + n_t) * ds_t(label_interf))
    assert abs(v) < 1.e-30
    v = assemble(dot(n_q + n_t, n_q + n_t) * ds_q(label_interf))
    assert abs(v) < 1.e-30


def test_submesh_assemble_quad_triangle():
    dim = 2
    label_ext = 1
    label_interf = 2
    mesh = Mesh(os.path.join(cwd, "..", "meshes", "mixed_cell_unit_square.msh"))
    mesh.topology_dm.markBoundaryFaces(dmcommon.FACE_SETS_LABEL, label_ext)
    mesh_t = Submesh(mesh, dim, PETSc.DM.PolytopeType.TRIANGLE, label_name="celltype", name="mesh_tri")
    x_t, y_t = SpatialCoordinate(mesh_t)
    n_t = FacetNormal(mesh_t)
    mesh_q = Submesh(mesh, dim, PETSc.DM.PolytopeType.QUADRILATERAL, label_name="celltype", name="mesh_quad")
    x_q, y_q = SpatialCoordinate(mesh_q)
    n_q = FacetNormal(mesh_q)
    V_t = FunctionSpace(mesh_t, "P", 4)
    V_q = FunctionSpace(mesh_q, "Q", 3)
    V = V_t * V_q
    u = TrialFunction(V)
    v = TestFunction(V)
    u_t, u_q = split(u)
    v_t, v_q = split(v)
    ds_t = Measure("ds", mesh_t, intersect_measures=(Measure("ds", mesh_q),))
    ds_q = Measure("ds", mesh_q, intersect_measures=(Measure("ds", mesh_t),))
    # Test against the base cases.
    c = x_t**2 * y_t**2
    a = c * inner(u_t, v_q) * ds_t(label_interf)
    A = assemble(a)
    c_ref = x_q**2 * y_q**2
    a_ref = c_ref * inner(TrialFunction(V_t), TestFunction(V_q)) * ds_t(label_interf)
    A_ref = assemble(a_ref)
    assert np.allclose(get_values(A, 1, 0), A_ref.M.values)
    c = x_t**2 * y_q**2
    a = c * inner(u_q, v_t) * ds_t(label_interf)
    A = assemble(a)
    c_ref = x_q**2 * y_t**2
    a_ref = c_ref * inner(TrialFunction(V_q), TestFunction(V_t)) * ds_t(label_interf)
    A_ref = assemble(a_ref)
    assert np.allclose(get_values(A, 0, 1), A_ref.M.values)
    c = dot(n_t, n_t)
    a = c * inner(u_t, v_q) * ds_q(label_interf)
    A = assemble(a)
    c_ref = dot(n_q, n_q)
    a_ref = c_ref * inner(TrialFunction(V_t), TestFunction(V_q)) * ds_q(label_interf)
    A_ref = assemble(a_ref)
    assert np.allclose(get_values(A, 1, 0), A_ref.M.values)
    c = dot(n_t, n_q)
    a = c * inner(u_q, v_t) * ds_q(label_interf)
    A = assemble(a)
    c_ref = dot(n_q, n_t)
    a_ref = c_ref * inner(TrialFunction(V_q), TestFunction(V_t)) * ds_q(label_interf)
    A_ref = assemble(a_ref)
    assert np.allclose(get_values(A, 0, 1), A_ref.M.values)
