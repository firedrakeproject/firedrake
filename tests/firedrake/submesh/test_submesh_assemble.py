import numpy as np
from firedrake import *


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
    dx0 = Measure("cell", domain=mesh)
    dx1 = Measure("cell", domain=subm)
    a = inner(u1, v0) * dx0(999) + inner(u0, v1) * dx1
    A = assemble(a, mat_type="nest")
    assert np.allclose(A.M.sparsity[0][0].nnz, [1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(A.M.sparsity[0][1].nnz, [4, 4, 4, 4, 0, 0])
    assert np.allclose(A.M.sparsity[1][0].nnz, [4, 4, 4, 4])
    assert np.allclose(A.M.sparsity[1][1].nnz, [1, 1, 1, 1])  # bc nodes
    M10 = np.array([[1./9. , 1./18., 1./36., 1./18., 0., 0.],   # noqa: E203
                    [1./18., 1./9. , 1./18., 1./36., 0., 0.],   # noqa: E203
                    [1./36., 1./18., 1./9. , 1./18., 0., 0.],   # noqa: E203
                    [1./18., 1./36., 1./18., 1./9. , 0., 0.]])  # noqa: E203
    assert np.allclose(A.M[0][1].values, np.transpose(M10))
    assert np.allclose(A.M[1][0].values, M10)


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
    dS0 = Measure("dS", domain=mesh)
    ds1 = Measure("ds", domain=subm)
    a = inner(u1, v0('+')) * dS0 + inner(u0('+'), v1) * ds1(5)
    A = assemble(a, mat_type="nest")
    assert np.allclose(A.M.sparsity[0][0].nnz, [1, 1, 1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(A.M.sparsity[0][1].nnz, [4, 4, 4, 4, 4, 4, 4, 4])
    assert np.allclose(A.M.sparsity[1][0].nnz, [8, 8, 8, 8])
    assert np.allclose(A.M.sparsity[1][1].nnz, [1, 1, 1, 1])  # bc nodes
    M10 = np.array([[0., 0., 0., 0., 0., 0., 1. / 3., 1. / 6.],
                    [0., 0., 0., 0., 0., 0., 1. / 6., 1. / 3.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.]])
    assert np.allclose(A.M[0][1].values, np.transpose(M10))
    assert np.allclose(A.M[1][0].values, M10)
    b = inner(u1, v0('+')) * ds1(5) + inner(u0('+'), v1) * dS0
    B = assemble(b, mat_type="nest")
    assert np.allclose(B.M.sparsity[0][0].nnz, [1, 1, 1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(B.M.sparsity[0][1].nnz, [4, 4, 4, 4, 4, 4, 4, 4])
    assert np.allclose(B.M.sparsity[1][0].nnz, [8, 8, 8, 8])
    assert np.allclose(B.M.sparsity[1][1].nnz, [1, 1, 1, 1])  # bc nodes
    assert np.allclose(B.M[0][1].values, A.M[0][1].values)
    assert np.allclose(B.M[1][0].values, A.M[1][0].values)


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
    dS = Measure("dS", domain=mesh)
    ds_l = Measure("ds", domain=mesh_l)
    ds_r = Measure("ds", domain=mesh_r)
    ds_rl = Measure("ds", domain=mesh_rl)
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
    assert np.allclose(A.M.sparsity[0][0].nnz, [1, 1, 1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(A.M.sparsity[0][1].nnz, [4, 4, 4, 4, 0, 0, 0, 0])
    assert np.allclose(A.M.sparsity[1][0].nnz, [4, 4, 4, 4])
    assert np.allclose(A.M.sparsity[1][1].nnz, [1, 1, 1, 1])  # bc nodes
    M10 = np.array([[0., 0., 1. / 3., 1. / 6., 0., 0., 0., 0.],
                    [0., 0., 1. / 6., 1. / 3., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.]])
    assert np.allclose(A.M[0][1].values, np.transpose(M10))
    assert np.allclose(A.M[1][0].values, M10)
    b = inner(u_rl, v_l) * dS(label_int) + inner(u_l, v_rl) * dS(label_int)
    B = assemble(b, mat_type="nest")
    assert np.allclose(B.M.sparsity[0][0].nnz, [1, 1, 1, 1, 1, 1, 1, 1])  # bc nodes
    assert np.allclose(B.M.sparsity[0][1].nnz, [4, 4, 4, 4, 0, 0, 0, 0])
    assert np.allclose(B.M.sparsity[1][0].nnz, [4, 4, 4, 4])
    assert np.allclose(B.M.sparsity[1][1].nnz, [1, 1, 1, 1])  # bc nodes
    assert np.allclose(B.M[0][1].values, A.M[0][1].values)
    assert np.allclose(B.M[1][0].values, A.M[1][0].values)


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
    dx = Measure("dx", domain=mesh)
    dx_l = Measure("dx", domain=mesh_l)
    dx_rl = Measure("dx", domain=mesh_rl)
    dS = Measure("dS", domain=mesh)
    dS_l = Measure("dS", domain=mesh_l)
    ds_rl = Measure("ds", domain=mesh_rl)
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
    ds_l = Measure("ds", domain=mesh_l)
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
    assert np.allclose(A.M.sparsity[0][0].nnz, [4, 4, 4, 4])
    assert np.allclose(A.M.sparsity[0][1].nnz, [4, 4, 4, 4])
    assert np.allclose(A.M.sparsity[1][0].nnz, [0, 0, 0, 0])
    assert np.allclose(A.M.sparsity[1][1].nnz, [1, 1, 1, 1])  # bc nodes
    M00 = np.array([[1. / 9. , 1. / 18., 1. / 36., 1. / 18.],  # noqa: E203
                    [1. / 18., 1. / 9. , 1. / 18., 1. / 36.],  # noqa: E203
                    [0., 0., 1. / 3., 1. / 6.],
                    [0., 0., 1. / 6., 1. / 3.]])
    M01 = np.array([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [- 1. / 6., - 1. / 3., 0., 0.],
                    [- 1. / 3., - 1. / 6., 0., 0.]])
    assert np.allclose(A.M[0][0].values, M00)
    assert np.allclose(A.M[0][1].values, M01)
