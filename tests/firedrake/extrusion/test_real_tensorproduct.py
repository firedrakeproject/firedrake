"""Tests for (foo x Real) exturded spaces"""
import numpy as np
import pytest

from firedrake import *


@pytest.fixture(params=["scalar", "vector", "tensor"])
def fs_kind(request):
    return request.param


@pytest.fixture
def V(extmesh, fs_kind):
    mesh = extmesh(2, 2, 8)
    fs = {
        "scalar": FunctionSpace,
        "vector": VectorFunctionSpace,
        "tensor": TensorFunctionSpace
    }
    return fs[fs_kind](mesh, "CG", 1, vfamily="Real", vdegree=0)


@pytest.fixture(params=["linear", "sin"])
def variant(request):
    return request.param


@pytest.fixture
def expr(variant, V, fs_kind):
    x, y, z = SpatialCoordinate(V.mesh())
    val = {"linear": z, "sin": sin(pi*z)}[variant]
    ret = {
        "scalar": val,
        "vector": as_vector((val, 0, 0)),
        "tensor": as_tensor(((val, 0, 0), (0, 0, 0), (0, 0, 0)))
    }
    return ret[fs_kind]


@pytest.fixture
def solution(variant, fs_kind):
    val = {"linear": 0.5, "sin": 2.0/pi}[variant]
    sol = {
        "scalar": val,
        "vector": as_vector((val, 0, 0)),
        "tensor": as_tensor(((val, 0, 0), (0, 0, 0), (0, 0, 0)))
    }
    return sol[fs_kind]


@pytest.fixture
def tolerance(variant):
    return {"linear": 2e-15, "sin": 1e-5}[variant]


def test_vertical_average(V, expr, solution, tolerance):
    """Test projection on P1 x Real space"""
    u = TrialFunction(V)
    v = TestFunction(V)

    out = Function(V)
    solve(inner(u, v)*dx == inner(expr, v)*dx, out)
    l2err = sqrt(assemble(inner(out-solution, out-solution)*dx))
    assert abs(l2err) < tolerance


@pytest.fixture(params=[False, True],
                ids=["prism", "hex"])
def quadrilateral(request):
    return request.param


def test_vertical_average_variable(quadrilateral):
    """Test computing vertical average on mesh with variable nb of levels"""
    tolerance = 2e-14
    mesh2d = RectangleMesh(5, 1, 5, 1, quadrilateral=quadrilateral)

    # construct number of levels
    xy = SpatialCoordinate(mesh2d)
    p0_2d = FunctionSpace(mesh2d, 'DG', 0)
    f_2d = Function(p0_2d).interpolate(1 + xy[0])
    max_layers = np.floor(f_2d.dat.data.real).astype(int)
    layers = np.zeros((p0_2d.dof_count, 2), dtype=int)
    layers[:, 1] = max_layers

    mesh = ExtrudedMesh(mesh2d, layers=layers, layer_height=1.0)

    fs = FunctionSpace(mesh, 'DG', 1)
    xyz = SpatialCoordinate(mesh)
    f = Function(fs).interpolate(xyz[2])

    p0 = FunctionSpace(mesh, 'DG', 0)
    correct = Function(p0, name='solution').interpolate(1 + xyz[0])
    correct.dat.data[:] = np.floor(correct.dat.data.real)/2

    fs_real = FunctionSpace(mesh, 'DG', 1, vfamily='Real', vdegree=0)
    f_real = Function(fs_real).project(f)

    l2err = sqrt(assemble(inner((f_real-correct), (f_real-correct))*dx))
    assert abs(l2err) < tolerance


@pytest.mark.parametrize(('testcase', 'tolerance'),
                         [(("CG", 1), 2e-7),
                          (("CG", 2), 1e-7),
                          (("CG", 3), 2e-7)],
                         ids=["CG1", "CG2", "CG3"])
def test_helmholtz(extmesh, quadrilateral, testcase, tolerance):
    """Solve depth-independent H. problem on Pn x Pn and Pn x Real spaces"""
    family, degree = testcase
    mesh = extmesh(2, 2, 8, quadrilateral=quadrilateral)

    fspace = FunctionSpace(mesh, family, degree, vfamily="Real", vdegree=0)
    refspace = FunctionSpace(mesh, family, degree,
                             vfamily=family, vdegree=degree)

    sol_real = Function(fspace)
    sol_ref = Function(refspace)
    for s in [sol_real, sol_ref]:
        fs = s.function_space()
        u = TrialFunction(fs)
        v = TestFunction(fs)

        f = Function(fs)
        xyz = SpatialCoordinate(mesh)
        f.interpolate((1+8*pi*pi)*cos(2*pi*xyz[0])*cos(2*pi*xyz[1]))

        solve(inner(grad(u), grad(v))*dx + inner(u, v)*dx == inner(f, v)*dx, s)

    l2err = sqrt(assemble(inner((sol_real-sol_ref), (sol_real-sol_ref))*dx))
    assert abs(l2err) < tolerance


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [(("CG", 1, (4, 6)), 1.9),
                          (("CG", 2, (3, 5)), 2.9),
                          (("CG", 3, (2, 4)), 3.9)],
                         ids=["CG1", "CG2", "CG3"])
def test_helmholtz_convergence(extmesh, quadrilateral, testcase, convrate):
    """Test convergence of depth-independent H. problem on Pn x Real space."""
    family, degree, (start, end) = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        mesh = extmesh(2**ii, 2**ii, 2**ii, quadrilateral=quadrilateral)

        fspace = FunctionSpace(mesh, family, degree, vfamily="Real", vdegree=0)

        u = TrialFunction(fspace)
        v = TestFunction(fspace)

        f = Function(fspace)
        xyz = SpatialCoordinate(mesh)
        f.interpolate((1+8*pi*pi)*cos(2*pi*xyz[0])*cos(2*pi*xyz[1]))

        out = Function(fspace)
        solve(inner(grad(u), grad(v))*dx + inner(u, v)*dx == inner(f, v)*dx, out)

        exact = Function(fspace)
        exact.interpolate(cos(2*pi*xyz[0])*cos(2*pi*xyz[1]))
        l2err[ii - start] = sqrt(assemble(inner((out-exact), (out-exact))*dx))
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()


def test_real_tensorproduct_mixed(V):
    mesh = V.mesh()
    Q = FunctionSpace(mesh, "P", 2)

    W = V*Q
    for (s_, s) in zip(W.subfunctions, (V, Q)):
        assert s_.node_set is s.node_set
        assert s_.dof_dset is s.dof_dset


def test_real_tensorproduct_component(V):
    for i in range(V.value_size):
        s = V.sub(i)
        assert V.node_set is s.node_set
