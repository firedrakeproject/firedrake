import pytest
from firedrake import *
try:
    import tinyasm  # noqa: F401
    marks = ()
except ImportError:
    marks = pytest.mark.skip(reason="No tinyasm")


@pytest.fixture(params=["Interval", "Triangle", "Quad"])
def mesh_type(request):
    return request.param


@pytest.fixture
def mesh(mesh_type):
    if mesh_type == "Interval":
        return ExtrudedMesh(IntervalMesh(10, 100), 5, 1/5)
    elif mesh_type == "Triangle":
        return ExtrudedMesh(RectangleMesh(10, 10, 100, 100), 5, 1/5)
    elif mesh_type == "Quad":
        return ExtrudedMesh(RectangleMesh(10, 10, 100, 100,
                                          quadrilateral=True), 5, 1/5)


@pytest.fixture
def S1family(mesh_type):
    if mesh_type == "Interval":
        return "CG"
    elif mesh_type == "Triangle":
        return "BDM"
    elif mesh_type == "Quad":
        return "RTCF"


@pytest.fixture
def expected(mesh_type):
    if mesh_type == "Interval":
        return [8, 13]
    elif mesh_type == "Triangle":
        return [11, 26]
    elif mesh_type == "Quad":
        return [9, 20]


@pytest.fixture(params=["petscasm", pytest.param("tinyasm", marks=marks)])
def backend(request):
    return request.param


@pytest.mark.skipcomplexnoslate
def test_linesmoother(mesh, S1family, expected, backend):
    nits = []
    for degree in range(2):
        S1 = FiniteElement(S1family, mesh._base_mesh.ufl_cell(), degree+1)
        S2 = FiniteElement("DG", mesh._base_mesh.ufl_cell(), degree)
        T0 = FiniteElement("CG", interval, degree+1)
        T1 = FiniteElement("DG", interval, degree)

        V2h_elt = HDiv(TensorProductElement(S1, T1))
        V2t_elt = TensorProductElement(S2, T0)
        V2v_elt = HDiv(V2t_elt)
        V3_elt = TensorProductElement(S2, T1)
        V2_elt = V2h_elt + V2v_elt

        V = FunctionSpace(mesh, V2_elt)
        Q = FunctionSpace(mesh, V3_elt)

        W = MixedFunctionSpace((V, Q))

        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        a = (inner(u, v) - inner(p, div(v))
             + inner(p, q) + inner(div(u), q))*dx
        bcs = [DirichletBC(W.sub(0), 0, "on_boundary"),
               DirichletBC(W.sub(0), 0, "top"),
               DirichletBC(W.sub(0), 0, "bottom")]

        x = SpatialCoordinate(mesh)
        if len(x) == 2:
            rsq = (x[0]-50)**2/20**2 + (x[1]-0.5)**2/0.2**2
        else:
            rsq = (x[0]-50)**2/20**2 + (x[1] - 50)**2/20**2 + (x[2]-0.5)**2/0.2**2
        f = exp(-rsq)

        L = inner(f, q)*dx

        w0 = Function(W)
        problem = LinearVariationalProblem(a, L, w0, bcs=bcs)

        wave_parameters = {'mat_type': 'matfree',
                           'ksp_type': 'preonly',
                           'pc_type': 'python',
                           'pc_python_type': 'firedrake.HybridizationPC',
                           'hybridization': {'ksp_type': 'cg',
                                             'ksp_monitor': None}}
        ls = {'pc_type': 'composite',
              'pc_composite_pcs': 'bjacobi,python',
              'pc_composite_type': 'additive',
              'sub_0': {'sub_pc_type': 'jacobi'},
              'sub_1': {'pc_type': 'python',
                        'pc_python_type': 'firedrake.ASMLinesmoothPC',
                        'pc_linesmooth_backend': backend,
                        'pc_linesmooth_codims': '0'}}

        wave_parameters['hybridization'].update(ls)

        solver = LinearVariationalSolver(problem, solver_parameters=wave_parameters)
        solver.solve()
        ctx = solver.snes.ksp.pc.getPythonContext()
        nits.append(ctx.trace_ksp.getIterationNumber())
    assert nits == expected
