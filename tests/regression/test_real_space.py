import pytest
import numpy as np

from firedrake import *


@pytest.mark.skipcomplex
def test_real_assembly():
    mesh = UnitIntervalMesh(3)
    fs = FunctionSpace(mesh, "Real", 0)
    f = Function(fs)

    f.dat.data[0] = 2.

    assert assemble(f * dx) == 2.0


@pytest.mark.skipcomplex
def test_real_one_form_assembly():
    mesh = UnitIntervalMesh(3)
    fs = FunctionSpace(mesh, "Real", 0)
    v = TestFunction(fs)

    assert assemble(v * dx).dat.data[0] == 1.0


@pytest.mark.skipcomplex
def test_real_two_form_assembly():
    mesh = UnitIntervalMesh(3)
    fs = FunctionSpace(mesh, "Real", 0)
    u = TrialFunction(fs)
    v = TestFunction(fs)

    assert assemble(2*u*v * dx).M.values == 2.0


@pytest.mark.skipcomplex
def test_real_nonsquare_two_form_assembly():
    mesh = UnitIntervalMesh(3)
    rfs = FunctionSpace(mesh, "Real", 0)
    cgfs = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(rfs)
    v = TestFunction(cgfs)

    base_case = assemble(2 * conj(v) * dx)
    m1 = assemble(2 * inner(u, v) * dx)

    u = TrialFunction(cgfs)
    v = TestFunction(rfs)
    m2 = assemble(2 * inner(u, v) * dx)

    np.testing.assert_almost_equal(base_case.dat.data,
                                   m1.M.values[:, 0])
    np.testing.assert_almost_equal(base_case.dat.data,
                                   m2.M.values[0, :])


@pytest.mark.skipcomplex
def test_real_mixed_one_form_assembly():
    mesh = UnitIntervalMesh(3)
    rfs = FunctionSpace(mesh, "Real", 0)
    cgfs = FunctionSpace(mesh, "CG", 1)

    mfs = cgfs*rfs
    v, q = TestFunctions(mfs)

    A = assemble(conj(v) * dx + q * dx)

    qq = TestFunction(rfs)

    AA = assemble(qq * dx)

    np.testing.assert_almost_equal(A.dat.data[1],
                                   AA.dat.data)


@pytest.mark.skipcomplex
def test_real_mixed_two_form_assembly():
    mesh = UnitIntervalMesh(3)
    rfs = FunctionSpace(mesh, "Real", 0)
    cgfs = FunctionSpace(mesh, "CG", 1)

    mfs = cgfs*rfs
    u, p = TrialFunctions(mfs)
    v, q = TestFunctions(mfs)

    m = assemble(inner(u, v) * dx + p * q * dx + u * q * dx + inner(p, v) * dx)

    qq = TestFunction(rfs)
    vv = TestFunction(cgfs)
    uu = TrialFunction(cgfs)

    m00 = assemble(inner(uu, vv) * dx)
    np.testing.assert_almost_equal(m00.M.values,
                                   m.M.blocks[0][0].values)
    m01 = assemble(uu * qq * dx)
    np.testing.assert_almost_equal(m01.M.values.T,
                                   m.M.blocks[0][1].values)
    np.testing.assert_almost_equal(m01.M.values,
                                   m.M.blocks[1][0].values)
    np.testing.assert_almost_equal(np.array([[1.]]),
                                   m.M.blocks[1][1].values)


@pytest.mark.skipcomplex
def test_real_mixed_monolithic_two_form_assembly():
    mesh = UnitIntervalMesh(3)
    rfs = FunctionSpace(mesh, "Real", 0)
    cgfs = FunctionSpace(mesh, "CG", 1)

    mfs = cgfs*rfs
    u, p = TrialFunctions(mfs)
    v, q = TestFunctions(mfs)

    with pytest.raises(ValueError):
        assemble(inner(u, v) * dx + p * q * dx + u * q * dx + inner(p, v) * dx, mat_type="aij")


@pytest.mark.skipcomplex
def test_real_extruded_mixed_two_form_assembly():
    m = UnitIntervalMesh(3)
    mesh = ExtrudedMesh(m, 10)
    rfs = FunctionSpace(mesh, "Real", 0)
    cgfs = FunctionSpace(mesh, "CG", 1)

    mfs = cgfs*rfs
    u, p = TrialFunctions(mfs)
    v, q = TestFunctions(mfs)

    m = assemble(inner(u, v) * dx + p * q * dx + u * q * dx + inner(p, v) * dx)

    qq = TestFunction(rfs)
    vv = TestFunction(cgfs)
    uu = TrialFunction(cgfs)

    m00 = assemble(inner(uu, vv) * dx)
    np.testing.assert_almost_equal(m00.M.values,
                                   m.M.blocks[0][0].values)
    m01 = assemble(uu * qq * dx)
    np.testing.assert_almost_equal(m01.M.values.T,
                                   m.M.blocks[0][1].values)
    np.testing.assert_almost_equal(m01.M.values,
                                   m.M.blocks[1][0].values)
    np.testing.assert_almost_equal(np.array([[1.]]),
                                   m.M.blocks[1][1].values)


@pytest.mark.skipcomplex
@pytest.mark.parallel
def test_real_mixed_solve():
    def poisson(resolution):
        mesh = IntervalMesh(resolution, pi)
        rfs = FunctionSpace(mesh, "Real", 0)
        cgfs = FunctionSpace(mesh, "CG", 1)

        mfs = cgfs*rfs

        v, q = TestFunctions(mfs)

        phi = Function(mfs)
        f = Function(mfs)
        x = SpatialCoordinate(mesh)

        f0, _ = f.split()

        f0.interpolate(cos(x[0]))

        f0, _ = split(f)

        phi_0, phi_1 = split(phi)

        residual_form = (inner(grad(phi_0), grad(v)) + inner(phi_1, v) + phi_0 * q - inner(f0, v)) * dx

        solve(residual_form == 0, phi)

        return sqrt(assemble(inner(f - phi, f - phi) * dx))

    assert ln(poisson(50)/poisson(100))/ln(2) > 1.99


@pytest.mark.skipcomplex
@pytest.mark.parallel
def test_real_mixed_solve_split_comms():
    def poisson(resolution):
        mesh = IntervalMesh(resolution, pi, comm=COMM_SELF)
        rfs = FunctionSpace(mesh, "Real", 0)
        cgfs = FunctionSpace(mesh, "CG", 1)

        mfs = cgfs*rfs

        v, q = TestFunctions(mfs)

        phi = Function(mfs)
        f = Function(mfs)
        x = SpatialCoordinate(mesh)

        f0, _ = f.split()

        f0.interpolate(cos(x[0]))

        f0, _ = split(f)

        phi_0, phi_1 = split(phi)

        residual_form = (inner(grad(phi_0), grad(v)) + inner(phi_1, v) + phi_0 * q - inner(f0, v)) * dx

        solve(residual_form == 0, phi)

        return sqrt(assemble(inner(f - phi, f - phi) * dx))

    assert ln(poisson(50)/poisson(100))/ln(2) > 1.99


@pytest.mark.skipcomplex
def test_real_space_eq():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "Real", 0)
    V2 = FunctionSpace(mesh, "Real", 0)
    assert V == V2
    assert V is not V2


@pytest.mark.skipcomplex
def test_real_space_mixed_assign():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "Real", 0)
    Q = FunctionSpace(mesh, "CG", 1)

    W = Q*V

    f = Function(W)

    q, v = f.split()

    q.assign(2)
    g = Function(V)

    g.dat.data[:] = 1
    v.assign(g)

    assert np.allclose(float(g), 1.0)
    assert np.allclose(float(g), float(v))
    assert np.allclose(q.dat.data_ro, 2.0)


@pytest.mark.skipcomplex
def test_real_space_first():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "Real", 0)
    Q = FunctionSpace(mesh, "CG", 1)
    MixedFunctionSpace([V, Q])


@pytest.mark.skipcomplex
def test_real_space_assign():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "Real", 0)
    f = Function(V)
    f.assign(2)
    g = Function(V)
    g.assign(2*f + f**3)
    h = Function(V)
    h.assign(0.0)
    assert np.allclose(float(f), 2.0)
    assert np.allclose(float(g), 12.0)
    assert np.allclose(float(h), 0.0)


@pytest.mark.skipcomplex
def test_real_interpolate():
    N = 100
    mesh = IntervalMesh(N, 0, 1)
    R = FunctionSpace(mesh, "R", 0)
    a_int = interpolate(Constant(1.0), R)
    assert np.allclose(float(a_int), 1.0)


@pytest.mark.skipcomplex
def test_real_interpolate_minmaxinc():
    N = 100
    mesh = IntervalMesh(N, 0, 1)
    R = FunctionSpace(mesh, "R", 0)
    x, = SpatialCoordinate(mesh)
    min_x = float(interpolate(x, R, access=MIN))
    max_x = float(interpolate(x, R, access=MAX))
    sum_x = float(interpolate(x, R, access=INC))

    # Midpoint evaluation in each cell.
    expect = np.linspace(0 + 1/(2*N), 1 - 1/(2*N), N)

    assert np.isclose(min_x, expect.min())
    assert np.isclose(max_x, expect.max())
    assert np.isclose(sum_x, expect.sum())
    min_x = Function(R).assign(-1)
    max_x = Function(R).assign(2)
    min_x = float(interpolate(x, min_x, access=MIN))
    max_x = float(interpolate(x, max_x, access=MAX))

    assert np.isclose(min_x, -1)
    assert np.isclose(max_x, 2)
