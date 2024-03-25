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
        return [5, 9]
    elif mesh_type == "Triangle":
        return [5, 11]
    elif mesh_type == "Quad":
        return [5, 11]


@pytest.fixture(params=["petscasm", pytest.param("tinyasm", marks=marks)])
def backend(request):
    return request.param


def test_linesmoother(mesh, S1family, expected, backend):
    base_cell = mesh._base_mesh.ufl_cell()
    S2family = "DG" if base_cell.is_simplex() else "DQ"
    DGfamily = "DG" if mesh.ufl_cell().is_simplex() else "DQ"
    nits = []
    for degree in range(2):
        S1 = FiniteElement(S1family, base_cell, degree+1)
        S2 = FiniteElement(S2family, base_cell, degree)
        T0 = FiniteElement("CG", interval, degree+1)
        T1 = FiniteElement("DG", interval, degree)

        V2h_elt = HDiv(TensorProductElement(S1, T1))
        V2t_elt = TensorProductElement(S2, T0)
        V2v_elt = HDiv(V2t_elt)
        V2_elt = V2h_elt + V2v_elt

        V = FunctionSpace(mesh, V2_elt)
        Q = FunctionSpace(mesh, DGfamily, degree, variant="integral")

        W = MixedFunctionSpace((V, Q))

        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        a = (-inner(u, v) * dx(degree=2*(degree+1))
             + (inner(p, div(v)) + inner(p + div(u), q)) * dx(degree=2*degree))

        gamma = Constant(1)
        aP = (inner(u, v) * dx(degree=2*(degree+1))
              + (inner(div(u) * gamma, div(v)) + inner(p * (1/gamma), q)) * dx(degree=2*degree))

        bcs = [DirichletBC(W.sub(0), 0, "on_boundary"),
               DirichletBC(W.sub(0), 0, "top"),
               DirichletBC(W.sub(0), 0, "bottom")]

        x = SpatialCoordinate(mesh)
        if len(x) == 2:
            rsq = (x[0]-50)**2/20**2 + (x[1]-0.5)**2/0.2**2
        else:
            rsq = (x[0]-50)**2/20**2 + (x[1] - 50)**2/20**2 + (x[2]-0.5)**2/0.2**2
        f = exp(-rsq)

        L = inner(f, q)*dx(degree=2*(degree+1))

        w0 = Function(W)
        problem = LinearVariationalProblem(a, L, w0, bcs=bcs, aP=aP, form_compiler_parameters={"mode": "vanilla"})

        wave_parameters = {'mat_type': 'matfree',
                           'pmat_type': 'nest',
                           'ksp_type': 'minres',
                           'ksp_monitor': None,
                           'ksp_norm_type': 'preconditioned',
                           'pc_type': 'fieldsplit',
                           'pc_fieldsplit_type': 'additive',
                           'fieldsplit_ksp_type': 'preonly',
                           'fieldsplit_0_pc_type': 'python',
                           'fieldsplit_0_pc_python_type': 'firedrake.ASMLinesmoothPC',
                           'fieldsplit_0_pc_linesmooth_backend': backend,
                           'fieldsplit_0_pc_linesmooth_codims': '0',
                           'fieldsplit_1_pc_type': 'jacobi'}

        solver = LinearVariationalSolver(problem, solver_parameters=wave_parameters)
        solver.solve()
        nits.append(solver.snes.ksp.getIterationNumber())
    assert nits == expected
