from firedrake import *


def test_linesmoother_vfs():

    H = 0.1
    mesh = ExtrudedMesh(IntervalMesh(10, 1), 5, H/5)
    V = VectorFunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(u, v) + inner(grad(u), grad(v)))*dx
    x, y = SpatialCoordinate(mesh)
    f = exp(-(x-0.5)**2-(y-0.5)**2)
    L = inner(f, v[0])*dx

    w = Function(V)
    problem = LinearVariationalProblem(a, L, w)

    test_parameters = {'ksp_type': 'cg',
                       'ksp_monitor': None,
                       'pc_type': 'composite',
                       'pc_composite_pcs': 'bjacobi,python',
                       'pc_composite_type': 'additive',
                       'sub_0': {'sub_pc_type': 'jacobi'},
                       'sub_1': {'pc_type': 'python',
                                 'pc_python_type': 'firedrake.ASMLinesmoothPC',
                                 'pc_linesmooth_codims': '1'}}

    solver = LinearVariationalSolver(problem,
                                     solver_parameters=test_parameters)
    solver.solve()
    nits = solver.snes.ksp.getIterationNumber()
    assert nits == 17
