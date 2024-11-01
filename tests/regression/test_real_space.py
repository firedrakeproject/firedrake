import pytest
import numpy as np

from firedrake import *
from firedrake.__future__ import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


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
def test_real_mixed_empty_component_assembly():
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    R = FunctionSpace(mesh, 'R', 0)
    W = V * R
    w = Function(W)
    v, _ = split(w)
    # This assembly has an empty block since the R component doesn't appear.
    # The test passes if the empty block doesn't cause the assembly to fail.
    assemble(derivative(inner(grad(v), grad(v)) * dx, w))


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


def mixed_poisson_opts():
    # The default R space solver options are sufficient if the direct solver is
    # MUMPS, otherwise we use an iterative solver to ensure the tests pass
    if DEFAULT_DIRECT_SOLVER == "mumps":
        opts = None
    else:
        opts = {
            "mat_type": "matfree",
            "ksp_type": "fgmres",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_0_fields": "0",
            "pc_fieldsplit_1_fields": "1",
            "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled": {
                    "ksp_type": "gmres",
                    "ksp_rtol": 1e-7,
                    "pc_type": "jacobi",
                },
            },
            "fieldsplit_1": {
                "ksp_type": "gmres",
                "pc_type": "none",
            }
        }
    return opts


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

        f0, _ = f.subfunctions

        f0.interpolate(cos(x[0]))

        f0, _ = split(f)

        phi_0, phi_1 = split(phi)

        residual_form = (inner(grad(phi_0), grad(v)) + inner(phi_1, v) + phi_0 * q - inner(f0, v)) * dx

        solve(residual_form == 0, phi, solver_parameters=mixed_poisson_opts())

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

        f0, _ = f.subfunctions

        f0.interpolate(cos(x[0]))

        f0, _ = split(f)

        phi_0, phi_1 = split(phi)

        residual_form = (inner(grad(phi_0), grad(v)) + inner(phi_1, v) + phi_0 * q - inner(f0, v)) * dx

        solve(residual_form == 0, phi, solver_parameters=mixed_poisson_opts())

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

    q, v = f.subfunctions

    q.assign(2)
    g = Function(V)

    g.dat.data[:] = 1
    v.assign(g)

    assert np.allclose(float(g), 1.0)
    assert np.allclose(float(g), float(v))
    assert np.allclose(q.dat.data_ro, 2.0)

    a = Function(W)
    b = Function(W).assign(2)

    with pytest.raises(ValueError):
        a.assign(b, subset="not None")

    a.assign(2*b)  # a = 2*2
    b += 3*a  # b = 2 + 3*4

    assert np.allclose(a.dat.split[0].data_ro, 4.0)
    assert np.allclose(a.dat.split[1].data_ro, 4.0)
    assert np.allclose(b.dat.split[0].data_ro, 14.0)
    assert np.allclose(b.dat.split[1].data_ro, 14.0)


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
    g.assign(2*f)
    h = Function(V)
    h.assign(0.0)
    assert np.allclose(float(f), 2.0)
    assert np.allclose(float(g), 4.0)
    assert np.allclose(float(h), 0.0)


@pytest.mark.skipcomplex
def test_real_interpolate():
    N = 100
    mesh = IntervalMesh(N, 0, 1)
    R = FunctionSpace(mesh, "R", 0)
    a_int = assemble(interpolate(Constant(1.0), R))
    assert np.allclose(float(a_int), 1.0)


def test_real_space_hex():
    mesh = BoxMesh(2, 1, 1, 2., 1., 1., hexahedral=True)
    DG = FunctionSpace(mesh, "DQ", 0)
    R = FunctionSpace(mesh, "R", 0)
    dg = Function(DG).assign(1.)
    r = Function(R).assign(1.)
    val = assemble(r * dx)
    assert abs(val - 2.) < 1.e-14
    val = assemble(inner(dg, TestFunction(R)) * dx)
    assert np.allclose(val.dat.data_ro, [2.])
    val = assemble(inner(r, TestFunction(DG)) * dx)
    assert np.allclose(val.dat.data, [1., 1.])
