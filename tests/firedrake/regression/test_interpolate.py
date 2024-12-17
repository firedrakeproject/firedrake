from os.path import abspath, dirname, join
import numpy as np
import pytest
from firedrake import *
from firedrake.__future__ import *

cwd = abspath(dirname(__file__))


def test_constant():
    cg1 = FunctionSpace(UnitSquareMesh(5, 5), "CG", 1)
    f = assemble(interpolate(Constant(1.0), cg1))
    assert np.allclose(1.0, f.dat.data)


def test_function():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    V1 = FunctionSpace(m, 'P', 1)
    V2 = FunctionSpace(m, 'P', 2)

    f = assemble(interpolate(x[0]*x[0], V1))
    g = assemble(interpolate(f, V2))

    # g shall be equivalent to:
    h = assemble(interpolate(x[0], V2))

    assert np.allclose(g.dat.data, h.dat.data)


def test_mixed_expression():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    V1 = FunctionSpace(m, 'P', 1)
    V2 = FunctionSpace(m, 'P', 2)

    V = V1 * V2
    expressions = [x[0], x[0]*x[1]]
    expr = as_vector(expressions)
    fg = assemble(interpolate(expr, V))
    f, g = fg.subfunctions

    f1 = Function(V1).interpolate(expressions[0])
    g1 = Function(V2).interpolate(expressions[1])
    assert np.allclose(f.dat.data, f1.dat.data)
    assert np.allclose(g.dat.data, g1.dat.data)


def test_mixed_function():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    V1 = FunctionSpace(m, 'RT', 1)
    V2 = FunctionSpace(m, 'DG', 0)
    V = V1 * V2

    expressions = [x[0], x[1], Constant(0.444)]
    expr = as_vector(expressions)
    v = assemble(interpolate(expr, V))

    W1 = FunctionSpace(m, 'RT', 2)
    W2 = FunctionSpace(m, 'DG', 1)
    W = W1 * W2
    w = assemble(interpolate(v, W))

    f, g = w.subfunctions
    f1 = Function(W1).interpolate(x)
    g1 = Function(W2).interpolate(expressions[-1])
    assert np.allclose(f.dat.data, f1.dat.data)
    assert np.allclose(g.dat.data, g1.dat.data)


def test_inner():
    m = UnitTriangleMesh()
    V1 = FunctionSpace(m, 'P', 1)
    V2 = FunctionSpace(m, 'P', 2)

    x, y = SpatialCoordinate(m)
    f = assemble(interpolate(inner(x, x), V1))
    g = assemble(interpolate(f, V2))

    # g shall be equivalent to:
    h = assemble(interpolate(x, V2))

    assert np.allclose(g.dat.data, h.dat.data)


def test_coordinates():
    cg2 = FunctionSpace(UnitSquareMesh(5, 5), "CG", 2)
    x = SpatialCoordinate(cg2.mesh())
    f = assemble(interpolate(x[0]*x[0], cg2))

    x = SpatialCoordinate(cg2.mesh())
    g = assemble(interpolate(x[0]*x[0], cg2))

    assert np.allclose(f.dat.data, g.dat.data)


def test_piola():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 1)
    V = FunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], Constant(0.0))), U)
    g = assemble(interpolate(f[0], V))

    # g shall be equivalent to:
    h = project(f[0], V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_vector():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 1)
    V = VectorFunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], Constant(0.0))), U)
    g = assemble(interpolate(f, V))

    # g shall be equivalent to:
    h = project(f, V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_tensor():
    mesh = UnitSquareMesh(2, 2)
    x = SpatialCoordinate(mesh)
    U = TensorFunctionSpace(mesh, 'P', 1)
    V = TensorFunctionSpace(mesh, 'CG', 2)

    c = as_tensor(((Constant(2.0), x[1]), (x[0], x[0] * x[1])))

    f = project(c, U)
    g = assemble(interpolate(f, V))

    # g shall be equivalent to:
    h = project(f, V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_constant_expression():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 1)
    V = FunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], x[1])), U)
    g = assemble(interpolate(div(f), V))

    assert np.allclose(2.0, g.dat.data)


def test_compound_expression():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 2)
    V = FunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], x[1])), U)
    g = assemble(interpolate(Constant(1.5)*div(f) + sin(x[0] * np.pi), V))

    # g shall be equivalent to:
    h = assemble(interpolate(3.0 + sin(pi * x[0]), V))

    assert np.allclose(g.dat.data, h.dat.data)


def test_hdiv_extruded_interval():
    mesh = ExtrudedMesh(UnitIntervalMesh(10), 10, 0.1)
    x = SpatialCoordinate(mesh)
    U = FunctionSpace(mesh, HDiv(TensorProductElement(FiniteElement("P", interval, 1), FiniteElement("DP", interval, 0))))
    expr = as_vector([x[0], x[1]])
    u = assemble(interpolate(expr, U))
    u_proj = project(expr, U)

    assert np.allclose(u.dat.data, u_proj.dat.data)


def test_hcurl_extruded_interval():
    mesh = ExtrudedMesh(UnitIntervalMesh(10), 10, 0.1)
    x = SpatialCoordinate(mesh)
    U = FunctionSpace(mesh, HCurl(TensorProductElement(FiniteElement("P", interval, 1), FiniteElement("DP", interval, 0))))
    expr = as_vector([x[0], x[1]])
    u = assemble(interpolate(expr, U))
    u_proj = project(expr, U)

    assert np.allclose(u.dat.data, u_proj.dat.data)


# Requires the relevant FInAT or FIAT duals to be defined
@pytest.mark.xfail(raises=NotImplementedError, reason="Requires the relevant FInAT or FIAT duals to be defined")
def test_hdiv_2d():
    mesh = UnitCubedSphereMesh(2)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    x = mesh.coordinates

    U = FunctionSpace(mesh, 'RTCF', 1)
    V = FunctionSpace(mesh, 'RTCF', 2)
    c = as_vector([x[1], -x[0], 0.0])

    f = project(c, U)
    g = assemble(interpolate(f, V))

    # g shall be equivalent to:
    h = project(f, V)

    assert np.allclose(g.dat.data, h.dat.data)


@pytest.mark.xfail(raises=NotImplementedError, reason="Requires the relevant FInAT or FIAT duals to be defined")
def test_hcurl_2d():
    mesh = UnitCubedSphereMesh(2)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    x = mesh.coordinates

    U = FunctionSpace(mesh, 'RTCE', 1)
    V = FunctionSpace(mesh, 'RTCE', 2)
    c = as_vector([-x[1], x[0], 0.0])

    f = project(c, U)
    g = assemble(interpolate(f, V))

    # g shall be equivalent to:
    h = project(f, V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_cell_orientation():
    m = UnitCubedSphereMesh(2)
    x = SpatialCoordinate(m)
    m.init_cell_orientations(x)
    x = m.coordinates
    U = FunctionSpace(m, 'RTCF', 1)
    V = VectorFunctionSpace(m, 'DQ', 1)

    f = project(as_tensor([x[1], -x[0], 0.0]), U)
    g = assemble(interpolate(f, V))

    # g shall be close to:
    h = project(f, V)

    assert abs(g.dat.data - h.dat.data).max() < 1e-2


def test_cell_orientation_curve():
    m = CircleManifoldMesh(3)
    x = SpatialCoordinate(m)
    m.init_cell_orientations(x)

    V = VectorFunctionSpace(m, 'DG', 0)
    f = assemble(interpolate(CellNormal(m), V))

    assert np.allclose(f.dat.data, [[1 / 2, sqrt(3) / 2],
                                    [-1, 0],
                                    [1 / 2, -sqrt(3) / 2]])


def test_cellvolume():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, 'DG', 0)

    f = assemble(interpolate(CellVolume(m), V))

    assert np.allclose(f.dat.data_ro, 0.125)


def test_cellvolume_higher_order_coords():
    m = UnitTriangleMesh()
    V = VectorFunctionSpace(m, 'P', 3)
    f = Function(V)
    f.interpolate(m.coordinates)

    # Warp mesh so that the bottom triangle line is:
    # x(x - 1)(x + a) with a = 19/12.0
    def warp(x):
        return x * (x - 1)*(x + 19/12.0)

    f.dat.data[1:3, 1] = warp(f.dat.data[1:3, 0])

    mesh = Mesh(f)
    g = assemble(interpolate(CellVolume(mesh), FunctionSpace(mesh, 'DG', 0)))

    assert np.allclose(g.dat.data_ro, 0.5 - (1.0/4.0 - (1 - 19.0/12.0)/3.0 - 19/24.0))


def test_mixed():
    m = UnitTriangleMesh()
    x = m.coordinates
    V1 = FunctionSpace(m, 'BDFM', 2)
    V2 = VectorFunctionSpace(m, 'P', 2)
    f = Function(V1 * V2)
    f.sub(0).project(as_tensor([x[1], -x[0]]))
    f.sub(1).interpolate(as_tensor([x[0], x[1]]))

    V = FunctionSpace(m, 'P', 1)
    g = assemble(interpolate(dot(grad(f[0]), grad(f[3])), V))

    assert np.allclose(1.0, g.dat.data)


def test_lvalue_rvalue():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.assign(1.0)
    u.interpolate(u + 1.0)
    assert np.allclose(u.dat.data_ro, 2.0)


@pytest.mark.parametrize("degree", range(1, 4))
def test_interpolator_Pk(degree):
    mesh = UnitSquareMesh(10, 10, quadrilateral=False)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", degree)
    P2 = FunctionSpace(mesh, "CG", degree + 1)

    expr = x[0]**degree + x[1]**degree
    x_P1 = assemble(interpolate(expr, P1))
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = assemble(interpolator.interpolate(x_P1))
    x_P2_direct = assemble(interpolate(expr, P2))

    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)


@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("variant", ("spectral", "equispaced"))
@pytest.mark.parametrize("quads", (True, False), ids=("quads", "triangles"))
def test_interpolator(quads, degree, variant):
    mesh = UnitSquareMesh(10, 10, quadrilateral=quads)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", degree, variant=variant)
    P2 = FunctionSpace(mesh, "CG", degree + 1)

    expr = x[0]**degree + x[1]**degree
    x_P1 = assemble(interpolate(expr, P1))
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = assemble(interpolator.interpolate(x_P1))
    x_P2_direct = assemble(interpolate(expr, P2))

    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)


def test_interpolator_tets():
    mesh = UnitTetrahedronMesh()
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    P2 = FunctionSpace(mesh, "CG", 2)

    expr = x[0] + x[1]
    x_P1 = assemble(interpolate(expr, P1))
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = assemble(interpolator.interpolate(x_P1))
    x_P2_direct = assemble(interpolate(expr, P2))

    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)


def test_interpolator_extruded():
    mesh = ExtrudedMesh(UnitSquareMesh(10, 10), 10, 0.1)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    P2 = FunctionSpace(mesh, "CG", 2)

    expr = x[0] + x[1]
    x_P1 = assemble(interpolate(expr, P1))
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = assemble(interpolator.interpolate(x_P1))
    x_P2_direct = assemble(interpolate(expr, P2))

    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)


def test_trace():
    mesh = UnitSquareMesh(10, 10)
    x = SpatialCoordinate(mesh)
    cg = FunctionSpace(mesh, "CG", 1)
    tr = FunctionSpace(mesh, "HDiv Trace", 1)

    expr = x[0] + x[1]
    x_cg = assemble(interpolate(expr, cg))
    x_tr_dir = assemble(interpolate(expr, tr))
    x_tr_cg = assemble(interpolate(x_cg, tr))

    assert np.allclose(x_tr_cg.dat.data, x_tr_dir.dat.data)


@pytest.mark.parametrize("degree", range(1, 4))
def test_adjoint_Pk(degree):
    mesh = UnitSquareMesh(10, 10)
    Pkp1 = FunctionSpace(mesh, "CG", degree+1)
    Pk = FunctionSpace(mesh, "CG", degree)

    v = conj(TestFunction(Pkp1))
    u_Pk = assemble(conj(TestFunction(Pk)) * dx)
    v_adj = assemble(interpolate(TestFunction(Pk), assemble(v * dx)))

    assert np.allclose(u_Pk.dat.data, v_adj.dat.data)


def test_adjoint_quads():
    mesh = UnitSquareMesh(10, 10)
    P1 = FunctionSpace(mesh, "CG", 1)
    P2 = FunctionSpace(mesh, "CG", 2)

    v = conj(TestFunction(P2))
    u_P1 = assemble(conj(TestFunction(P1)) * dx)
    v_adj = assemble(interpolate(TestFunction(P1), assemble(v * dx)))

    assert np.allclose(u_P1.dat.data, v_adj.dat.data)


def test_adjoint_dg():
    mesh = UnitSquareMesh(10, 10)
    cg1 = FunctionSpace(mesh, "CG", 1)
    dg1 = FunctionSpace(mesh, "DG", 1)

    v = conj(TestFunction(dg1))
    u_cg = assemble(conj(TestFunction(cg1)) * dx)
    v_adj = assemble(interpolate(TestFunction(cg1), assemble(v * dx)))

    assert np.allclose(u_cg.dat.data, v_adj.dat.data)


@pytest.mark.parametrize("degree", range(1, 4))
def test_function_cofunction(degree):
    mesh = UnitSquareMesh(10, 10)
    Pkp1 = FunctionSpace(mesh, "CG", degree+1)
    Pk = FunctionSpace(mesh, "CG", degree)

    v1 = conj(TestFunction(Pkp1))
    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(sin(2*pi*x[0])*sin(2*pi*x[1]), Pk))

    fhat = assemble(f*v1*dx)
    norm_i = assemble(interpolate(f, fhat))
    norm = assemble(f*f*dx)

    assert np.allclose(norm_i, norm)


@pytest.mark.skipcomplex  # complex numbers are not orderable
def test_interpolate_periodic_coords_max():
    mesh = PeriodicUnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "P", 1)

    continuous = assemble(interpolate(SpatialCoordinate(mesh), V, access=MAX))

    # All nodes on the "seam" end up being 1, not 0.
    assert np.allclose(np.unique(continuous.dat.data_ro.round(decimals=16)),
                       [0.25, 0.5, 0.75, 1])


def test_basic_dual_eval_cg3():
    mesh = UnitIntervalMesh(1)
    V = FunctionSpace(mesh, "CG", 3)
    x = SpatialCoordinate(mesh)
    expr = Constant(1.)
    f = assemble(interpolate(expr, V))
    assert np.allclose(f.dat.data_ro[f.cell_node_map().values], [node(expr) for node in f.function_space().finat_element.fiat_equivalent.dual_basis()])
    expr = x[0]**3
    # Account for cell and corresponding expression being flipped onto
    # reference cell before reaching FIAT
    expr_fiat = (1-x[0])**3
    f = assemble(interpolate(expr, V))
    assert np.allclose(f.dat.data_ro[f.cell_node_map().values], [node(expr_fiat) for node in f.function_space().finat_element.fiat_equivalent.dual_basis()])


def test_basic_dual_eval_bdm():
    mesh = UnitTriangleMesh()
    V = FunctionSpace(mesh, "BDM", 2)
    x = SpatialCoordinate(mesh)
    expr = as_vector([x[0], x[1]])
    f = assemble(interpolate(expr, V))
    dual_basis = f.function_space().finat_element.fiat_equivalent.dual_basis()
    # Can't do nodal evaluation of the FIAT dual basis yet so just check the
    # dat is the correct length
    assert len(f.dat.data_ro) == len(dual_basis)


def test_quadrature():
    from ufl.geometry import QuadratureWeight
    mesh = UnitIntervalMesh(1)
    Qse = FiniteElement("Quadrature", mesh.ufl_cell(), degree=2, quad_scheme="default")
    Qs = FunctionSpace(mesh, Qse)
    fiat_rule = Qs.finat_element.fiat_equivalent
    # For spatial coordinate we should get 2 points per cell
    x, = SpatialCoordinate(mesh)
    # Account for cell and corresponding expression being flipped onto
    # reference cell before reaching FIAT
    expr_fiat = 1-x
    xq = assemble(interpolate(expr_fiat, Qs))
    assert np.allclose(xq.dat.data_ro[xq.cell_node_map().values].T, fiat_rule._points)
    # For quadrature weight we should 2 equal weights for each cell
    w = QuadratureWeight(mesh)
    wq = assemble(interpolate(w, Qs))
    assert np.allclose(wq.dat.data_ro[wq.cell_node_map().values].T, fiat_rule._weights)


def test_interpolation_tensor_convergence():
    errors = []
    for n in range(2, 9):
        mesh = UnitSquareMesh(2**n, 2**n)
        # ||expr - I(expr)||_L2 = c h^k for degree k
        V = TensorFunctionSpace(mesh, "RT", 1)
        x, y = SpatialCoordinate(mesh)

        vs = V.value_shape
        expr = as_tensor(np.asarray([
            sin(2*pi*x*(i+1))*cos(4*pi*y*i)
            for i in range(np.prod(vs, dtype=int))
        ], dtype=object).reshape(vs))

        f = assemble(interpolate(expr, V))

        errors.append(norm(expr - f))

    errors = np.asarray(errors)

    rate = np.log2(errors[:-1] / errors[1:])
    assert (rate[-2:] > 0.98).all()


def test_interpolation_tensor_symmetric():
    mesh = UnitSquareMesh(8, 7)
    # Interpolation of a symmetric tensor should be the same whether
    # we have symmetry or not.
    V = TensorFunctionSpace(mesh, "RT", 1, symmetry=True)
    Vexp = TensorFunctionSpace(mesh, "RT", 1)
    J = Jacobian(mesh)
    K = JacobianInverse(mesh)
    # Make a symmetric tensor-valued expression
    expr = as_tensor([J*J.T, K*K.T])
    expr = as_tensor(expr[i, j, k], (j, k, i))
    f = assemble(interpolate(expr, V))
    fexp = assemble(interpolate(expr, Vexp))
    assert np.isclose(norm(fexp - f), 0)


@pytest.mark.parallel(nprocs=3)
def test_interpolation_on_hex():
    # "cube_hex.msh" contains all possible facet orientations.
    meshfile = join(cwd, "..", "meshes", "cube_hex.msh")
    mesh = Mesh(meshfile)
    p = 4
    V = FunctionSpace(mesh, "Q", p)
    x, y, z = SpatialCoordinate(mesh)
    expr = x**p * y**p * z**p
    f = Function(V).interpolate(expr)
    assert assemble((f - expr)**2 * dx) < 1e-13
    assert abs(assemble(f * dx) - 1./(p + 1)**3) < 1e-11


def test_interpolate_logical_not():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 1)
    x, y = SpatialCoordinate(mesh)

    a = assemble(interpolate(conditional(Not(x < .2), 1, 0), V))
    b = assemble(interpolate(conditional(x >= .2, 1, 0), V))
    assert np.allclose(a.dat.data, b.dat.data)
