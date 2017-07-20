import pytest
import numpy as np
from firedrake import *


def run_vector_valued_test(x, degree=1, family='RT'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(m, family, degree)
    expr = ['cos(x[0]*pi*2)*sin(x[1]*pi*2)']*2
    e = Expression(expr)
    exact = Function(VectorFunctionSpace(m, 'CG', 5))
    exact.interpolate(e)

    # Solve to machine precision.
    ret = project(e, V, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble(inner((ret - exact), (ret - exact)) * dx))


def run_vector_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = VectorFunctionSpace(m, family, degree)
    expr = ['cos(x[0]*pi*2)*sin(x[1]*pi*2)']*2
    e = Expression(expr)
    exact = Function(VectorFunctionSpace(m, 'CG', 5))
    exact.interpolate(e)

    # Solve to machine precision.  This version of the test uses the
    # alternate syntax in which the target Function is already
    # available.
    ret = Function(V)
    project(e, ret, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble(inner((ret - exact), (ret - exact)) * dx))


def run_tensor_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = TensorFunctionSpace(m, family, degree)
    expr = [['cos(x[0]*pi*2)*sin(x[1]*pi*2)', 'cos(x[0]*pi*2)*sin(x[1]*pi*2)'],
            ['cos(x[0]*pi*2)*sin(x[1]*pi*2)', 'cos(x[0]*pi*2)*sin(x[1]*pi*2)']]
    e = Expression(expr)
    exact = Function(TensorFunctionSpace(m, 'CG', 5))
    exact.interpolate(e)

    # Solve to machine precision.  This version of the test uses the
    # alternate syntax in which the target Function is already
    # available.
    ret = Function(V)
    project(e, ret, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble(inner((ret - exact), (ret - exact)) * dx))


def run_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(m, family, degree)
    e = Expression('cos(x[0]*pi*2)*sin(x[1]*pi*2)')
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
    ev = Expression('x[0]')
    eu = Expression(('x[0]', 'x[1]'))
    with pytest.raises(RuntimeError):
        project(v, U)
    with pytest.raises(RuntimeError):
        project(u, V)
    with pytest.raises(RuntimeError):
        project(ev, U)
    with pytest.raises(RuntimeError):
        project(eu, V)


def test_project_mismatched_mesh():
    m2 = UnitSquareMesh(2, 2)
    m3 = UnitCubeMesh(2, 2, 2)

    U = FunctionSpace(m2, 'CG', 1)
    V = FunctionSpace(m3, 'CG', 1)

    u = Function(U)
    v = Function(V)

    with pytest.raises(RuntimeError):
        project(u, V)

    with pytest.raises(RuntimeError):
        project(v, U)


def test_project_mismatched_shape():
    m = UnitSquareMesh(2, 2)

    U = VectorFunctionSpace(m, 'CG', 1, dim=3)
    V = VectorFunctionSpace(m, 'CG', 1, dim=2)

    u = Function(U)
    v = Function(V)

    with pytest.raises(RuntimeError):
        project(u, V)

    with pytest.raises(RuntimeError):
        project(v, U)


def test_repeatable():
    mesh = UnitSquareMesh(1, 1)
    Q = FunctionSpace(mesh, 'DG', 1)

    V2 = FunctionSpace(mesh, 'DG', 0)
    V3 = FunctionSpace(mesh, 'DG', 0)
    W = V2 * V3
    expr = Expression('1.0')
    old = project(expr, Q)

    f = project(Expression(('-1.0', '-1.0')), W)  # noqa
    new = project(expr, Q)

    for fd, ud in zip(new.dat.data, old.dat.data):
        assert (fd == ud).all()


def test_projector():
    m = UnitSquareMesh(2, 2)
    Vc = FunctionSpace(m, "CG", 2)
    v = Function(Vc).interpolate(Expression("x[0]*x[1] + cos(x[0]+x[1])"))
    mass1 = assemble(v*dx)

    Vd = FunctionSpace(m, "DG", 1)
    vo = Function(Vd)

    P = Projector(v, vo)
    P.project()

    mass2 = assemble(vo*dx)
    assert(np.abs(mass1-mass2) < 1.0e-10)

    v.interpolate(Expression("x[1] + exp(x[0]+x[1])"))
    mass1 = assemble(v*dx)

    P.project()
    mass2 = assemble(vo*dx)
    assert(np.abs(mass1-mass2) < 1.0e-10)


def test_trivial_projector():
    m = UnitSquareMesh(2, 2)
    Vc = FunctionSpace(m, "CG", 2)
    v = Function(Vc).interpolate(Expression("x[0]*x[1] + cos(x[0]+x[1])"))
    mass1 = assemble(v*dx)

    vo = Function(Vc)

    P = Projector(v, vo)
    P.project()

    mass2 = assemble(vo*dx)
    assert(np.abs(mass1-mass2) < 1.0e-10)

    v.interpolate(Expression("x[1] + exp(x[0]+x[1])"))
    mass1 = assemble(v*dx)

    P.project()
    mass2 = assemble(vo*dx)
    assert(np.abs(mass1-mass2) < 1.0e-10)


def test_projector_expression():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    vo = Function(V)
    expr = Expression("1")
    with pytest.raises(ValueError):
        Projector(expr, vo)


@pytest.mark.parametrize('tensor', ['scalar', 'vector', 'tensor'])
@pytest.mark.parametrize('same_fspace', [False, True])
def test_projector_bcs(tensor, same_fspace):
    mesh = UnitSquareMesh(4, 4)
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

    assert errornorm(ret, ref) < 1.0e-13


@pytest.mark.parametrize(('family', 'degree', 'quad'),
                         [('RT', 1, False), ('BDM', 1, False), ('RTCF', 1, True),
                          ('RT', 2, False), ('BDM', 2, False), ('RTCF', 2, True)])
def test_average_method(family, degree, quad):
    mesh = UnitSquareMesh(4, 4, quadrilateral=quad)
    if quad:
        element = FiniteElement(family, quadrilateral, degree)
    else:
        element = FiniteElement(family, triangle, degree)

    V = FunctionSpace(mesh, element)
    x = SpatialCoordinate(mesh)
    vo = Function(V).project(as_vector([x[0] ** 2, x[1] ** 2]))

    V_d = FunctionSpace(mesh, BrokenElement(element))
    vd = Function(V_d).project(vo)

    v_rec = project(vd, V, method="average")
    assert errornorm(v_rec, vo) < 1.0e-13

    v_c = Function(V)
    project(vd, v_c, method="average")
    assert errornorm(v_c, vo) < 1.0e-13


@pytest.mark.parametrize(('family', 'degree', 'quad'),
                         [('RT', 1, False), ('BDM', 1, False), ('RTCF', 1, True),
                          ('RT', 2, False), ('BDM', 2, False), ('RTCF', 2, True)])
def test_average_method_sphere_domain(family, degree, quad):
    if quad:
        element = FiniteElement(family, quadrilateral, degree)
        mesh = UnitCubedSphereMesh(refinement_level=3)
    else:
        element = FiniteElement(family, triangle, degree)
        mesh = UnitIcosahedralSphereMesh(refinement_level=3)

    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    V = FunctionSpace(mesh, element)
    vo = Function(V).project(as_vector([x[0] ** 2, x[1] ** 2, x[2] ** 2]))

    V_d = FunctionSpace(mesh, BrokenElement(element))
    vd = Function(V_d).project(vo)

    v_rec = project(vd, V, method="average")
    assert errornorm(v_rec, vo) < 1.0e-13

    v_c = Function(V)
    project(vd, v_c, method="average")
    assert errornorm(v_c, vo) < 1.0e-13


@pytest.mark.parametrize(('family', 'degree', 'tensor'),
                         [('CG', 1, 'vector'), ('CG', 2, 'vector'), ('CG', 3, 'vector'),
                          ('Q', 1, 'vector'), ('Q', 2, 'vector'), ('Q', 3, 'vector'),
                          ('CG', 1, 'tensor'), ('CG', 2, 'tensor'), ('CG', 3, 'tensor'),
                          ('Q', 1, 'tensor'), ('Q', 2, 'tensor'), ('Q', 3, 'tensor')])
def test_average_method_dg_to_cg(family, degree, tensor):
    if family == 'Q':
        mesh = UnitSquareMesh(4, 4, quadrilateral=True)
        if rank == 'vector':
            Vc = VectorFunctionSpace(mesh, family, degree)
            Vdg = VectorFunctionSpace(mesh, "DQ", degree)
            x = SpatialCoordinate(mesh)
            vo = Function(Vc).project(as_vector([x[0] ** 2, x[1] ** 2]))

        else:
            Vc = TensorFunctionSpace(mesh, family, degree)
            Vdg = TensorFunctionSpace(mesh, "DQ", degree)
            x = SpatialCoordinate(mesh)
            vo = Function(Vc).project(as_tensor([[x[0] ** 2, x[1] ** 2],
                                                 [x[0] ** 2, x[1] ** 2]]))

    else:
        mesh = UnitSquareMesh(4, 4)
        if rank == 'vector':
            Vc = VectorFunctionSpace(mesh, family, degree)
            Vdg = VectorFunctionSpace(mesh, "DG", degree)
            x = SpatialCoordinate(mesh)
            vo = Function(Vc).project(as_vector([x[0] ** 2, x[1] ** 2]))

        else:
            Vc = TensorFunctionSpace(mesh, family, degree)
            Vdg = TensorFunctionSpace(mesh, "DG", degree)
            x = SpatialCoordinate(mesh)
            vo = Function(Vc).project(as_tensor([[x[0] ** 2, x[1] ** 2],
                                                 [x[0] ** 2, x[1] ** 2]]))

    vd = project(vo, Vdg)

    v_rec = project(vd, Vc, method="average")
    assert errornorm(v_rec, vo) < 1.0e-13

    v_c = Function(Vc)
    project(vd, v_c, method="average")
    assert errornorm(v_c, vo) < 1.0e-13


@pytest.mark.parametrize('tensor', ['scalar', 'vector', 'tensor'])
def test_averaging_bcs(tensor):
    mesh = UnitSquareMesh(4, 4)
    x = SpatialCoordinate(mesh)
    if tensor == 'scalar':
        Vcg = FunctionSpace(mesh, "CG", 1)
        Vdg = FunctionSpace(mesh, "DG", 1)
        bcs = [DirichletBC(Vcg, Constant(0.5), (1, 3)),
               DirichletBC(Vcg, Constant(-0.5), (2, 4))]
        fct = cos(x[0]*pi*2)*sin(x[1]*pi*2)

    elif tensor == 'vector':
        Vcg = VectorFunctionSpace(mesh, "CG", 1)
        Vdg = VectorFunctionSpace(mesh, "DG", 1)
        bcs = [DirichletBC(Vcg, Constant((0.5, 0.5)), (1, 3)),
               DirichletBC(Vcg, Constant((-0.5, -0.5)), (2, 4))]
        fct = as_vector([cos(x[0]*pi*2)*sin(x[1]*pi*2),
                         cos(x[0]*pi*2)*sin(x[1]*pi*2)])

    elif tensor == 'tensor':
        Vcg = TensorFunctionSpace(mesh, "CG", 1)
        Vdg = TensorFunctionSpace(mesh, "DG", 1)
        bcs = [DirichletBC(Vcg, Constant(((0.5, 0.5),
                                          (0.5, 0.5))), (1, 3)),
               DirichletBC(Vcg, Constant(((-0.5, -0.5),
                                          (-0.5, -0.5))), (2, 4))]
        fct = as_tensor([[cos(x[0]*pi*2)*sin(x[1]*pi*2),
                          cos(x[0]*pi*2)*sin(x[1]*pi*2)],
                         [cos(x[0]*pi*2)*sin(x[1]*pi*2),
                          cos(x[0]*pi*2)*sin(x[1]*pi*2)]])

    vcg = project(fct, Vcg, bcs=bcs)

    vdg = project(vcg, Vdg)
    v = project(vdg, Vcg, method="average")

    assert errornorm(v, vcg) < 1.0e-13


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
