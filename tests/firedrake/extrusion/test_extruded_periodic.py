import pytest
from firedrake import *
from firedrake.utils import ScalarType
from firedrake.mesh import make_mesh_from_coordinates
import numpy as np


def test_extruded_periodic_facet_integrals():
    # +---+---+
    # | 5.|13.|
    # +---+---+
    # | 3.|11.|
    # +---+---+
    # | 2.| 7.|
    # +---+---+
    mesh = RectangleMesh(2, 1, 2.0, 1.0, quadrilateral=True)
    extm = ExtrudedMesh(mesh, layers=3, layer_height=1.0, extrusion_type="uniform", periodic=True)
    x, y, z = SpatialCoordinate(extm)
    V = FunctionSpace(extm, "DG", 0)
    f = Function(V).interpolate(conditional(And(x < 1.0, z < 1.0), 2.0,
                                conditional(And(x < 1.0, z < 2.0), 3.0,
                                conditional(And(x < 1.0, z < 3.0), 5.0,
                                conditional(And(x < 2.0, z < 1.0), 7.0,  # noqa: E128
                                conditional(And(x < 2.0, z < 2.0), 11.0, 13.0))))))  # noqa: E128
    val = assemble(f('-')*dS_h)
    assert abs(val - 41.) < 1.e-12
    val = assemble(f('+')*dS_h)
    assert abs(val - 41.) < 1.e-12
    val = assemble(f('-') * dS_v)
    assert abs(val - 31.) < 1.e-12
    val = assemble(f('+') * dS_v)
    assert abs(val - 10.) < 1.e-12
    val = assemble(f * ds_v(1))
    assert abs(val - 10.) < 1.e-12
    val = assemble(f * ds_v(2))
    assert abs(val - 31.) < 1.e-12
    val = assemble(f * ds_v(3))
    assert abs(val - 41.) < 1.e-12
    val = assemble(f * ds_v(4))
    assert abs(val - 41.) < 1.e-12
    val = assemble(f * ds_b)
    assert abs(val - 9.) < 1.e-12
    val = assemble(f * ds_t)
    assert abs(val - 18.) < 1.e-12


def test_extruded_periodic_boundary_nodes():
    # +-----+------+
    # |     |      |
    # 7  3  11 15 19
    # |     |      |
    # 6--2--10-14-18
    # |     |      |
    # 5  1  9  13 17
    # |     |      |
    # 4--0--8--12-16
    mesh = UnitIntervalMesh(2)
    extm = ExtrudedMesh(mesh, layers=2, extrusion_type="uniform", periodic=True)
    V = FunctionSpace(extm, "CG", 2)
    assert (V.boundary_nodes(1) == np.array([4, 5, 6, 7])).all()
    assert (V.boundary_nodes(2) == np.array([16, 17, 18, 19])).all()
    assert (V.boundary_nodes("bottom") == np.array([0, 4, 8, 12, 16])).all()
    with pytest.raises(ValueError):
        assert V.boundary_nodes("top")


def test_extruded_periodic_1_layer():
    # Test coner cases.
    mesh = UnitIntervalMesh(1)
    extm = ExtrudedMesh(mesh, layers=1, extrusion_type="uniform", periodic=True)
    # DG0 x CG1: +---+
    #            |   |
    #            +-0-+
    # Identified as DG0 x DG0
    elem = TensorProductElement(FiniteElement("DG", mesh.ufl_cell(), 0),
                                FiniteElement("CG", "interval", 1))
    V = FunctionSpace(extm, elem)
    v = TestFunction(V)
    u = TrialFunction(V)
    A = assemble(inner(u, v) * dx)
    assert np.allclose(A.M.values, np.array([[1.]], dtype=ScalarType))
    # DG0 x CG2: +---+
    #            | 1 |
    #            +-0-+
    elem = TensorProductElement(FiniteElement("DG", mesh.ufl_cell(), 0),
                                FiniteElement("CG", "interval", 2))
    V = FunctionSpace(extm, elem)
    v = TestFunction(V)
    u = TrialFunction(V)
    A = assemble(inner(u, v) * dx)
    assert np.allclose(A.M.values, np.array([[1. / 5., 2. / 15.], [2. / 15., 8. / 15]], dtype=ScalarType))


@pytest.mark.parallel(nprocs=3)
def test_extruded_periodic_poisson():
    n = 64
    mesh = UnitIntervalMesh(n)
    extm = ExtrudedMesh(mesh, layers=n, extrusion_type="uniform", periodic=True)
    V = FunctionSpace(extm, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(extm)
    f.interpolate(- 8.0 * pi * pi * cos(2 * pi * x) * cos(2 * pi * y))
    a = - inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx
    exact = Function(V).interpolate(cos(2 * pi * x) * cos(2 * pi * y))
    bc = DirichletBC(V, exact, (1, 2))
    sol = Function(V)
    solve(a == L, sol, bcs=[bc])
    assert sqrt(assemble(inner(sol - exact, sol - exact) * dx)) < 1.e-7


@pytest.mark.parallel(nprocs=3)
def test_extruded_periodic_annulus():
    m = 5  # num. element in radial direction
    n = 7  # num. element in circumferential direction
    # Make reference mesh: mesh0
    mesh = CircleManifoldMesh(n)
    mesh0 = ExtrudedMesh(mesh, layers=m, layer_height=1.0 / m, extrusion_type="radial")
    # Make test mesh: mesh1
    mesh = IntervalMesh(m, 1.0, 2.0)
    mesh1 = ExtrudedMesh(mesh, layers=n, layer_height=2 * pi / n, extrusion_type="uniform", periodic=True)
    elem1 = mesh1.coordinates.ufl_element()
    coordV1 = FunctionSpace(mesh1, elem1)
    x1, y1 = SpatialCoordinate(mesh1)
    coord1 = Function(coordV1).interpolate(as_vector([x1 * cos(y1), x1 * sin(y1)]))
    mesh1 = make_mesh_from_coordinates(coord1.topological, "annulus")
    mesh1._base_mesh = mesh
    # Check volume
    x0, y0 = SpatialCoordinate(mesh0)
    x1, y1 = SpatialCoordinate(mesh1)
    vol0 = assemble(Constant(1) * dx(domain=mesh0))
    vol1 = assemble(Constant(1) * dx(domain=mesh1))
    assert abs(vol1 - vol0) < 1.e-12
    # Check projection
    RTCF0 = FunctionSpace(mesh0, "RTCF", 3)
    RTCF1 = FunctionSpace(mesh1, "RTCF", 3)
    f0 = Function(RTCF0).project(as_vector([sin(x0) + 2.0, cos(y0) + 3.0]), solver_parameters={"ksp_rtol": 1.e-13})
    f1 = Function(RTCF1).project(as_vector([sin(x1) + 2.0, cos(y1) + 3.0]), solver_parameters={"ksp_rtol": 1.e-13})
    int0 = assemble(inner(f0, as_vector([x0 + 5.0, y0 + 7.0])) * dx)
    int1 = assemble(inner(f1, as_vector([x1 + 5.0, y1 + 7.0])) * dx)
    assert abs(int1 - int0) < 1.e-12
    # Check mixed poisson
    inner_boun_id0 = "bottom"
    outer_boun_id0 = "top"
    inner_boun_id1 = 1
    outer_boun_id1 = 2
    # -- reference
    DG0 = FunctionSpace(mesh0, "DG", 2)
    W0 = RTCF0 * DG0
    u0, p0 = TrialFunctions(W0)
    v0, q0 = TestFunctions(W0)
    u0_ = f0
    p0_ = Function(DG0).interpolate(x0 + y0 + 1.0)
    a0 = (inner(u0, v0) + inner(p0, div(v0)) + inner(div(u0), q0)) * dx
    L0 = conj(q0) * dx
    bcs0 = [DirichletBC(W0.sub(0), u0_, outer_boun_id0),
            DirichletBC(W0.sub(1), p0_, inner_boun_id0)]
    w0 = Function(W0)
    solve(a0 == L0, w0, bcs=bcs0)
    # -- test
    DG1 = FunctionSpace(mesh1, "DG", 2)
    W1 = RTCF1 * DG1
    u1, p1 = TrialFunctions(W1)
    v1, q1 = TestFunctions(W1)
    u1_ = f1
    p1_ = Function(DG1).interpolate(x1 + y1 + 1.0)
    a1 = (inner(u1, v1) + inner(p1, div(v1)) + inner(div(u1), q1)) * dx
    L1 = conj(q1) * dx
    bcs1 = [DirichletBC(W1.sub(0), u1_, outer_boun_id1),
            DirichletBC(W1.sub(1), p1_, inner_boun_id1)]
    w1 = Function(W1)
    solve(a1 == L1, w1, bcs=bcs1)
    # -- Check solutions
    uint0 = assemble(inner(w0.sub(0), as_vector([sin(x0) + 0.2, cos(y0) + 0.3])) * dx)
    uint1 = assemble(inner(w1.sub(0), as_vector([sin(x1) + 0.2, cos(y1) + 0.3])) * dx)
    assert abs(uint1 - uint0) < 1.e-12
    pint0 = assemble(inner(w0.sub(1), x0 * y0 + 2.0) * dx)
    pint1 = assemble(inner(w1.sub(1), x1 * y1 + 2.0) * dx)
    assert abs(pint1 - pint0) < 1.e-12
