import pytest
import numpy as np
from firedrake import *


def run_vector_valued_test(x, degree=1, family='RT'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(m, family, degree)
    xs = SpatialCoordinate(m)
    expr = cos(xs[0]*pi*2)*sin(xs[1]*pi*2)
    expr = as_vector([expr, expr])
    exact = Function(VectorFunctionSpace(m, 'CG', 5))
    exact.interpolate(expr)

    # Solve to machine precision.
    ret = project(expr, V, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble(inner((ret - exact), (ret - exact)) * dx))


def run_vector_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = VectorFunctionSpace(m, family, degree)
    xs = SpatialCoordinate(m)

    expr = cos(xs[0]*pi*2)*sin(xs[1]*pi*2)
    expr = as_vector([expr, expr])
    exact = Function(VectorFunctionSpace(m, 'CG', 5))
    exact.interpolate(expr)

    # Solve to machine precision.  This version of the test uses the
    # alternate syntax in which the target Function is already
    # available.
    ret = Function(V)
    project(expr, ret, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble(inner((ret - exact), (ret - exact)) * dx))


def run_tensor_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = TensorFunctionSpace(m, family, degree)
    xs = SpatialCoordinate(m)
    expr = as_matrix([[cos(xs[0]*pi*2)*sin(xs[1]*pi*2), cos(xs[0]*pi*2)*sin(xs[1]*pi*2)],
                      [cos(xs[0]*pi*2)*sin(xs[1]*pi*2), cos(xs[0]*pi*2)*sin(xs[1]*pi*2)]])
    exact = Function(TensorFunctionSpace(m, 'CG', 5))
    exact.interpolate(expr)

    # Solve to machine precision.  This version of the test uses the
    # alternate syntax in which the target Function is already
    # available.
    ret = Function(V)
    project(expr, ret, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble(inner((ret - exact), (ret - exact)) * dx))


def run_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(m, family, degree)
    xs = SpatialCoordinate(m)
    e = cos(xs[0]*pi*2)*sin(xs[1]*pi*2)
    exact = Function(FunctionSpace(m, 'CG', 5))
    exact.interpolate(e)

    # Solve to machine precision. This version of the test uses the
    # method version of project.
    ret = Function(V)
    ret.project(e, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble((ret - exact) * (ret - exact) * dx))


@pytest.mark.parametrize(('degree', 'family', 'expected_convergence'), [
    (1, 'CG', 1.8),
    (2, 'CG', 2.6),
    (3, 'CG', 3.8),
    (0, 'DG', 0.8),
    (1, 'DG', 1.8),
    (2, 'DG', 2.8)])
def test_convergence(degree, family, expected_convergence):
    l2_diff = np.array([run_test(x, degree, family) for x in range(2, 5)])
    conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    assert (conv > expected_convergence).all()


@pytest.mark.parametrize(('degree', 'family', 'expected_convergence'), [
    (1, 'CG', 1.8),
    (2, 'CG', 2.6),
    (3, 'CG', 3.8),
    (0, 'DG', 0.8),
    (1, 'DG', 1.8),
    (2, 'DG', 2.8)])
def test_vector_convergence(degree, family, expected_convergence):
    l2_diff = np.array([run_vector_test(x, degree, family) for x in range(2, 5)])
    conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    assert (conv > expected_convergence).all()


@pytest.mark.parametrize(('degree', 'family', 'expected_convergence'), [
    (1, 'CG', 1.8),
    (2, 'CG', 2.6),
    (3, 'CG', 3.8),
    (0, 'DG', 0.8),
    (1, 'DG', 1.8),
    (2, 'DG', 2.8)])
def test_tensor_convergence(degree, family, expected_convergence):
    l2_diff = np.array([run_tensor_test(x, degree, family) for x in range(2, 5)])
    conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    assert (conv > expected_convergence).all()


@pytest.mark.parametrize(('degree', 'family', 'expected_convergence'), [
    (1, 'RT', 0.75),
    (2, 'RT', 1.8),
    (3, 'RT', 2.8),
    (1, 'BDM', 1.8),
    (2, 'BDM', 2.8),
    (3, 'BDM', 3.8)])
def test_vector_valued_convergence(degree, family, expected_convergence):
    l2_diff = np.array([run_vector_valued_test(x, degree, family)
                        for x in range(2, 6)])
    conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    assert (conv > expected_convergence).all()


def test_project_mismatched_rank():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, 'CG', 1)
    U = FunctionSpace(m, 'RT', 1)
    v = Function(V)
    u = Function(U)
    xs = SpatialCoordinate(m)
    ev = xs[0]
    eu = as_vector((xs[0], xs[1]))
    with pytest.raises(ValueError):
        project(v, U)
    with pytest.raises(ValueError):
        project(u, V)
    with pytest.raises(ValueError):
        project(ev, U)
    with pytest.raises(ValueError):
        project(eu, V)


def test_project_mismatched_mesh():
    m2 = UnitSquareMesh(2, 2)
    m3 = UnitCubeMesh(2, 2, 2)

    U = FunctionSpace(m2, 'CG', 1)
    V = FunctionSpace(m3, 'CG', 1)

    u = Function(U)
    v = Function(V)

    with pytest.raises(ValueError):
        project(u, V)

    with pytest.raises(ValueError):
        project(v, U)


def test_project_mismatched_shape():
    m = UnitSquareMesh(2, 2)

    U = VectorFunctionSpace(m, 'CG', 1, dim=3)
    V = VectorFunctionSpace(m, 'CG', 1, dim=2)

    u = Function(U)
    v = Function(V)

    with pytest.raises(ValueError):
        project(u, V)

    with pytest.raises(ValueError):
        project(v, U)


def test_repeatable():
    mesh = UnitSquareMesh(1, 1)
    Q = FunctionSpace(mesh, 'DG', 1)

    V2 = FunctionSpace(mesh, 'DG', 0)
    V3 = FunctionSpace(mesh, 'DG', 0)
    W = V2 * V3
    expr = Constant(1.0)
    old = project(expr, Q)

    f = project(as_vector((-1.0, -1.0)), W)  # noqa
    new = project(expr, Q)

    for fd, ud in zip(new.dat.data, old.dat.data):
        assert (fd == ud).all()


@pytest.mark.parametrize('mat_type', ['aij', 'matfree'])
def test_projector(mat_type):
    m = UnitSquareMesh(2, 2)
    Vc = FunctionSpace(m, "CG", 2)
    xs = SpatialCoordinate(m)

    v = Function(Vc).interpolate(xs[0]*xs[1] + cos(xs[0]+xs[1]))
    mass1 = assemble(v*dx)

    Vd = FunctionSpace(m, "DG", 1)
    vo = Function(Vd)

    P = Projector(v, vo, solver_parameters={"mat_type": mat_type})
    P.project()

    mass2 = assemble(vo*dx)
    assert np.abs(mass1-mass2) < 1.0e-10

    v.interpolate(xs[1] + exp(xs[0]+xs[1]))
    mass1 = assemble(v*dx)

    P.project()
    mass2 = assemble(vo*dx)
    assert np.abs(mass1-mass2) < 1.0e-10


@pytest.mark.parametrize('mat_type', ['aij', 'nest', 'matfree'])
def test_mixed_projector(mat_type):
    m = UnitSquareMesh(2, 2)
    Vc1 = FunctionSpace(m, "CG", 1)
    Vc2 = FunctionSpace(m, "CG", 2)
    Vc = Vc1 * Vc2
    xs = SpatialCoordinate(m)

    v = Function(Vc)
    v0, v1 = v.subfunctions
    v0.interpolate(xs[0]*xs[1] + cos(xs[0]+xs[1]))
    v1.interpolate(xs[0]*xs[1] + sin(xs[0]+xs[1]))
    mass1 = assemble(sum(split(v))*dx)

    Vd1 = FunctionSpace(m, "DG", 1)
    Vd2 = FunctionSpace(m, "DG", 2)
    Vd = Vd1 * Vd2
    vo = Function(Vd)

    P = Projector(v, vo, solver_parameters={"mat_type": mat_type})
    P.project()

    mass2 = assemble(sum(split(vo))*dx)
    assert np.abs(mass1-mass2) < 1.0e-10

    v0.interpolate(xs[1] + exp(xs[0]+xs[1]))
    v1.interpolate(xs[0] + exp(xs[0]+xs[1]))
    mass1 = assemble(sum(split(v))*dx)

    P.project()
    mass2 = assemble(sum(split(vo))*dx)
    assert np.abs(mass1-mass2) < 1.0e-10


def test_trivial_projector():
    m = UnitSquareMesh(2, 2)
    Vc = FunctionSpace(m, "CG", 2)
    xs = SpatialCoordinate(m)

    v = Function(Vc).interpolate(xs[0]*xs[1] + cos(xs[0]+xs[1]))
    mass1 = assemble(v*dx)

    vo = Function(Vc)

    P = Projector(v, vo)
    P.project()

    mass2 = assemble(vo*dx)
    assert np.abs(mass1-mass2) < 1.0e-10

    v.interpolate(xs[1] + exp(xs[0]+xs[1]))
    mass1 = assemble(v*dx)

    P.project()
    mass2 = assemble(vo*dx)
    assert np.abs(mass1-mass2) < 1.0e-10


@pytest.mark.parametrize('tensor', ['scalar', 'vector', 'tensor'])
@pytest.mark.parametrize('same_fspace', [False, True])
def test_projector_bcs(tensor, same_fspace):
    mesh = UnitSquareMesh(2, 2)
    x = SpatialCoordinate(mesh)
    if tensor == 'scalar':
        V = FunctionSpace(mesh, "CG", 1)
        V_ho = FunctionSpace(mesh, "CG", 5)
        bcs = [DirichletBC(V_ho, Constant(0.5), (1, 3)),
               DirichletBC(V_ho, Constant(-0.5), (2, 4))]
        fct = cos(x[0]*pi*2)*sin(x[1]*pi*2)

    elif tensor == 'vector':
        V = VectorFunctionSpace(mesh, "CG", 1)
        V_ho = VectorFunctionSpace(mesh, "CG", 5)
        bcs = [DirichletBC(V_ho, Constant((0.5, 0.5)), (1, 3)),
               DirichletBC(V_ho, Constant((-0.5, -0.5)), (2, 4))]
        fct = as_vector([cos(x[0]*pi*2)*sin(x[1]*pi*2),
                         cos(x[0]*pi*2)*sin(x[1]*pi*2)])

    elif tensor == 'tensor':
        V = TensorFunctionSpace(mesh, "CG", 1)
        V_ho = TensorFunctionSpace(mesh, "CG", 5)
        bcs = [DirichletBC(V_ho, Constant(((0.5, 0.5),
                                           (0.5, 0.5))), (1, 3)),
               DirichletBC(V_ho, Constant(((-0.5, -0.5),
                                           (-0.5, -0.5))), (2, 4))]
        fct = as_tensor([[cos(x[0]*pi*2)*sin(x[1]*pi*2),
                          cos(x[0]*pi*2)*sin(x[1]*pi*2)],
                         [cos(x[0]*pi*2)*sin(x[1]*pi*2),
                          cos(x[0]*pi*2)*sin(x[1]*pi*2)]])

    if same_fspace:
        v = Function(V_ho).project(fct)
    else:
        v = Function(V).project(fct)

    ret = Function(V_ho)
    projector = Projector(v, ret, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                                              "pc_type": "lu"})
    projector.project()

    # Manually solve a Galerkin projection problem to get a reference
    ref = Function(V_ho)
    p = TrialFunction(V_ho)
    q = TestFunction(V_ho)
    a = inner(p, q)*dx
    L = inner(v, q)*dx
    solve(a == L, ref, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                                   "pc_type": "lu"})

    assert errornorm(ret, ref) < 1.0e-10


@pytest.mark.parametrize(('degree', 'family', 'expected_convergence'), [
    (0, 'DGT', 0.8),
    (1, 'DGT', 1.8),
    (2, 'DGT', 2.8)])
def test_DGT_convergence(degree, family, expected_convergence):
    l2_diff = np.array([run_trace_projection(x, degree, family) for x in range(2, 5)])
    conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    assert (conv > expected_convergence).all()

    # I decreased the mesh param x here because it is a 3D problem and x=5 was running quite long
    l2_diff = np.array([run_extr_trace_projection(x, degree, family) for x in range(1, 4)])
    conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    assert (conv > expected_convergence).all()


def run_trace_projection(x, degree=1, family='DGT'):
    m = UnitSquareMesh(2 ** x, 2 ** x, quadrilateral=False)
    x = SpatialCoordinate(m)
    f = x[0]*(2-x[0])*x[1]*(2-x[1])

    V_ho = FunctionSpace(m, 'CG', 6)
    ref = Function(V_ho).interpolate(f)

    T = FunctionSpace(m, family, degree)
    w = Function(T)
    w.project(f, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    area = FacetArea(m)
    return sqrt(assemble(area * inner((w - ref), (w - ref)) * ds
                         + area * inner((w('+') - ref('+')), (w('+') - ref('+'))) * dS))


def run_extr_trace_projection(x, degree=1, family='DGT'):
    base = UnitSquareMesh(2 ** x, 2 ** x, quadrilateral=False)
    m = ExtrudedMesh(base, 2 ** x)
    x = SpatialCoordinate(m)
    f = x[0]*(2-x[0])*x[1]*(2-x[1])*x[2]*(2-x[2])

    V_ho = FunctionSpace(m, 'CG', 6)
    ref = Function(V_ho).interpolate(f)

    T = FunctionSpace(m, family, degree=degree)
    w = Function(T)
    w.project(f, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    area = FacetArea(m)
    return sqrt(assemble(area * (w - ref) * (w - ref) * ds_v
                         + area * (w - ref) * (w - ref) * ds_t
                         + area * (w - ref) * (w - ref) * ds_b
                         + area * (w('+') - ref('+')) * (w('+') - ref('+')) * dS_h
                         + area * (w('+') - ref('+')) * (w('+') - ref('+')) * dS_v))
