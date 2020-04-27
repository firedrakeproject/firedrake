import pytest
from firedrake import *


@pytest.fixture(params=[1, 2, 3],
                ids=["Extruded Interval", "Extruded Triangle Mesh", "Extruded Quadrilateral Mesh"])
def mesh(request):
    layers = 5
    layer_height = 1/layers
    if request.param == 1:
        m = IntervalMesh(10, 100)
        S1family = "CG"
        S2family = "DG"
        degree_shift = 1
        cell = interval
    if request.param == 2:
        m = RectangleMesh(10, 10, 100, 100)
        S1family = "BDM"
        S2family = "DG"
        degree_shift = 1
        cell = triangle
    if request.param == 3:
        m = RectangleMesh(10, 10, 100, 100, quadrilateral = True)
        S1family = "RTCF"
        S2family = "DG"
        degree_shift = 1
        cell = quadrilateral
    mex = ExtrudedMesh(m, layers, layer_height)
    return {'mesh':mex, 'layers':layers, 'S1family':S1family, 'S2family':S2family,
            'degree_shift':degree_shift, 'cell':cell}

@pytest.fixture
def expected(mesh):
    if mesh["S1family"] == "CG":
        return [8, 13]
    elif mesh["S1family"] == "BDM":
        return [10, 26]
    elif mesh["S1family"] == "RTCF":
        return [9, 20]


def test_linesmoother(mesh, expected):

    nits = []
    for degree in range(2):
        S1 = FiniteElement(mesh["S1family"], mesh["cell"], degree+mesh["degree_shift"])
        S2 = FiniteElement(mesh["S2family"], mesh["cell"], degree)
        T0 = FiniteElement("CG", interval, degree+1)
        T1 = FiniteElement("DG", interval, degree)

        V2h_elt = HDiv(TensorProductElement(S1, T1))
        V2t_elt = TensorProductElement(S2, T0)
        V2v_elt = HDiv(V2t_elt)
        V3_elt = TensorProductElement(S2, T1)
        V2_elt = V2h_elt + V2v_elt

        V = FunctionSpace(mesh["mesh"], V2_elt)
        Q = FunctionSpace(mesh["mesh"], V3_elt)
        
        W = MixedFunctionSpace((V, Q))

        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        a = (inner(v,u) - div(v)*p + p*q + div(u)*q)*dx
        bcs = [DirichletBC(W.sub(0), 0, "on_boundary"),
               DirichletBC(W.sub(0), 0, "top"),
               DirichletBC(W.sub(0), 0, "bottom")]

        x = SpatialCoordinate(mesh["mesh"])
        if len(x) == 2:
            rsq = (x[0]-50)**2/20**2 + (x[1]-0.5)**2/0.2**2
        else:
            rsq = (x[0]-50)**2/20**2 + (x[1] - 50)**2/20**2 + (x[2]-0.5)**2/0.2**2
        f = exp(-rsq)

        L = q*f*dx

        w0 = Function(W)
        problem = LinearVariationalProblem(a, L, w0, bcs=bcs)

        wave_parameters = {'mat_type': 'matfree',
                           'ksp_type': 'preonly',
                           'pc_type': 'python',
                           'pc_python_type': 'firedrake.HybridizationPC',
                           'hybridization': {'ksp_type': 'cg',
                                             'ksp_max_it': expected[degree],
                                             'ksp_monitor':None}}
        ls = {  'pc_type': 'composite',
                'pc_composite_pcs': 'bjacobi,python',
                'pc_composite_type': 'additive',
                'sub_0': {'sub_pc_type': 'jacobi'},
                'sub_1': {  'pc_type': 'python',
                            'pc_python_type': 'firedrake.ASMLinesmoothPC',
                            'pc_asm_codims': '0'}}

        wave_parameters['hybridization'].update(ls)
        
        solver = LinearVariationalSolver(problem, solver_parameters=wave_parameters)
        solver.solve()
        ctx = solver.snes.ksp.pc.getPythonContext()
        print(ctx.trace_ksp)
        nits.append(ctx.trace_ksp.getIterationNumber())
    assert (nits == expected)
