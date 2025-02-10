import pytest

from enum import Enum, auto
from numpy.testing import assert_allclose
import numpy as np
try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
    PETSc = None
try:
    import slepc4py.SLEPc as SLEPc
except ModuleNotFoundError:
    SLEPc = None
from firedrake import *
from firedrake.adjoint import *
from pyadjoint import Block, MinimizationProblem, TAOSolver, get_working_tape
from pyadjoint.optimization.tao_solver import OptionsManager, PETScVecInterface


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    if annotate_tape():
        pause_annotation()


@pytest.mark.skipcomplex
def test_petsc_roundtrip_single():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space_1 = FunctionSpace(mesh, "Lagrange", 1)
    space_2 = FunctionSpace(mesh, "Lagrange", 2)

    u_1 = Function(space_1).interpolate(X[0])
    u_2 = Function(space_2).interpolate(-1 - X[0])

    for m in [u_1, u_2]:
        vec_interface = PETScVecInterface(m)
        x = vec_interface.new_petsc()
        m_test = Function(m.function_space())
        vec_interface.to_petsc(x, m)
        vec_interface.from_petsc(x, m_test)
        assert (m.dat.data_ro == m_test.dat.data_ro).all()


@pytest.mark.skipcomplex
def test_petsc_roundtrip_multiple():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space_1 = FunctionSpace(mesh, "Lagrange", 1)
    space_2 = FunctionSpace(mesh, "Lagrange", 2)

    u_1 = Function(space_1).interpolate(X[0])
    u_2 = Function(space_2).interpolate(-1 - X[0])

    vec_interface = PETScVecInterface((u_1, u_2))
    x = vec_interface.new_petsc()
    u_1_test = Function(space_1)
    u_2_test = Function(space_2)
    vec_interface.to_petsc(x, (u_1, u_2))
    vec_interface.from_petsc(x, (u_1_test, u_2_test))
    assert (u_1.dat.data_ro == u_1_test.dat.data_ro).all()
    assert (u_2.dat.data_ro == u_2_test.dat.data_ro).all()


def minimize_tao_lmvm(rf, *, convert_options=None):
    problem = MinimizationProblem(rf)
    solver = TAOSolver(problem, {"tao_type": "lmvm",
                                 "tao_gatol": 1.0e-7,
                                 "tao_grtol": 0.0,
                                 "tao_gttol": 0.0},
                       convert_options=convert_options)
    return solver.solve()


def minimize_tao_nls(rf, *, convert_options=None):
    problem = MinimizationProblem(rf)
    solver = TAOSolver(problem, {"tao_type": "nls",
                                 "tao_gatol": 1.0e-7,
                                 "tao_grtol": 0.0,
                                 "tao_gttol": 0.0},
                       convert_options=convert_options)
    return solver.solve()


@pytest.mark.parametrize("minimize", [minimize,
                                      minimize_tao_lmvm,
                                      minimize_tao_nls])
@pytest.mark.skipcomplex
def test_optimisation_constant_control(minimize):
    """This tests a list of controls in a minimisation"""
    mesh = UnitSquareMesh(1, 1)
    R = FunctionSpace(mesh, "R", 0)

    n = 3
    x = [Function(R) for i in range(n)]
    c = [Control(xi) for xi in x]

    # Rosenbrock function https://en.wikipedia.org/wiki/Rosenbrock_function
    # with minimum at x = (1, 1, 1, ...)
    f = sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(n-1))

    J = assemble(f * dx(domain=mesh))
    rf = ReducedFunctional(J, c)
    result = minimize(rf)
    assert_allclose([float(xi) for xi in result], 1., rtol=1e-4)


def _simple_helmholz_model(V, source):
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(v), grad(u))*dx + 100.0*v*u*dx - v*source*dx
    solve(F == 0, u)
    return u


@pytest.mark.skipcomplex
def test_simple_inversion():
    """Test inversion of source term in helmholze eqn."""
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    source_ref = Function(V)
    x = SpatialCoordinate(mesh)
    source_ref.interpolate(cos(pi*x**2))

    # compute reference solution
    with stop_annotating():
        u_ref = _simple_helmholz_model(V, source_ref)

    # now rerun annotated model with zero source
    source = Function(V)
    c = Control(source)
    u = _simple_helmholz_model(V, source)

    J = assemble(1e6 * (u - u_ref)**2*dx)
    rf = ReducedFunctional(J, c)

    x = minimize(rf)
    assert_allclose(x.dat.data, source_ref.dat.data, rtol=1e-2)
    rf(source)
    x = minimize(rf, derivative_options={"riesz_representation": "l2"})
    assert_allclose(x.dat.data, source_ref.dat.data, rtol=1e-2)
    rf(source)
    x = minimize(rf, derivative_options={"riesz_representation": "H1"})
    # Assert that the optimisation does not converge for H1 representation
    assert not np.allclose(x.dat.data, source_ref.dat.data, rtol=1e-2)


@pytest.mark.parametrize("minimize", [minimize_tao_lmvm,
                                      minimize_tao_nls])
@pytest.mark.parametrize("riesz_representation", [None, "l2", "L2", "H1"])
@pytest.mark.skipcomplex
def test_tao_simple_inversion(minimize, riesz_representation):
    """Test inversion of source term in helmholze eqn using TAO."""
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    source_ref = Function(V)
    x = SpatialCoordinate(mesh)
    source_ref.interpolate(cos(pi*x**2))

    # compute reference solution
    with stop_annotating():
        u_ref = _simple_helmholz_model(V, source_ref)

    # now rerun annotated model with zero source
    source = Function(V)
    c = Control(source)
    u = _simple_helmholz_model(V, source)

    J = assemble(1e6 * (u - u_ref)**2*dx)
    rf = ReducedFunctional(J, c)

    x = minimize(rf, convert_options=(None if riesz_representation is None
                                      else {"riesz_representation": riesz_representation}))
    assert_allclose(x.dat.data, source_ref.dat.data, rtol=1e-2)


class TransformType(Enum):
    PRIMAL = auto()
    DUAL = auto()


def transform(v, transform_type, *args, mfn_parameters=None, **kwargs):
    with stop_annotating():
        if mfn_parameters is None:
            mfn_parameters = {}
        mfn_parameters = dict(mfn_parameters)

        space = v.function_space()
        if not ufl.duals.is_primal(space):
            space = space.dual()
        if not ufl.duals.is_primal(space):
            raise NotImplementedError("Mixed primal/dual space case not implemented")
        comm = v.comm

        class M:
            def mult(self, A, x, y):
                if transform_type == TransformType.PRIMAL:
                    v = Cofunction(space.dual())
                elif transform_type == TransformType.DUAL:
                    v = Function(space)
                else:
                    raise ValueError(f"Unrecognized transform_type: {transform_type}")
                with v.dat.vec_wo as v_v:
                    x.copy(result=v_v)
                u = v.riesz_representation(*args, **kwargs)
                with u.dat.vec_ro as u_v:
                    u_v.copy(result=y)

        with v.dat.vec_ro as v_v:
            n, N = v_v.getSizes()
        M_mat = PETSc.Mat().createPython(((n, N), (n, N)),
                                         M(), comm=comm)
        M_mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        M_mat.setUp()

        mfn = SLEPc.MFN().create(comm=comm)
        options = OptionsManager(mfn_parameters, None)
        options.set_default_parameter("fn_type", "sqrt")
        mfn.setOperator(M_mat)

        options.set_from_options(mfn)
        mfn.setUp()
        if mfn.getFN().getType() != SLEPc.FN.Type.SQRT:
            raise ValueError("Invalid FN type")

        with v.dat.vec_ro as v_v:
            x = v_v.copy()
            y = v_v.copy()

        if y.norm(PETSc.NormType.NORM_INFINITY) == 0:
            x.zeroEntries()
        else:
            mfn.solve(y, x)
            if mfn.getConvergedReason() <= 0:
                raise RuntimeError("Convergence failure")

        if ufl.duals.is_primal(v):
            u = Function(space)
        else:
            u = Cofunction(space.dual())
        with u.dat.vec_wo as u_v:
            x.copy(result=u_v)

    if annotate_tape():
        block = TransformBlock(v, transform_type, *args, mfn_parameters=mfn_parameters, **kwargs)
        block.add_output(u.block_variable)
        get_working_tape().add_block(block)

    return u


class TransformBlock(Block):
    def __init__(self, v, *args, **kwargs):
        super().__init__()
        self.add_dependency(v)
        self._args = args
        self._kwargs = kwargs

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        lam, = adj_inputs
        return transform(lam, *self._args, **self._kwargs)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        v, = inputs
        return transform(v, *self._args, **self._kwargs)


@pytest.mark.skipif(SLEPc is None, reason="SLEPc not available")
@pytest.mark.parametrize("tao_type", ["lmvm",
                                      "blmvm"])
@pytest.mark.skipcomplex
def test_simple_inversion_riesz_representation(tao_type):
    """Test use of a Riesz map in inversion of source term in helmholze eqn
    using TAO."""

    riesz_representation = "L2"
    mfn_parameters = {"mfn_type": "krylov",
                      "mfn_tol": 1.0e-12}
    tao_parameters = {"tao_type": tao_type,
                      "tao_gatol": 1.0e-5,
                      "tao_grtol": 0.0,
                      "tao_gttol": 0.0,
                      "tao_monitor": None}

    with stop_annotating():
        mesh = UnitIntervalMesh(10)
        V = FunctionSpace(mesh, "CG", 1)
        source_ref = Function(V)
        x = SpatialCoordinate(mesh)
        source_ref.interpolate(cos(pi*x**2))
        u_ref = _simple_helmholz_model(V, source_ref)

    def forward(source):
        c = Control(source)
        u = _simple_helmholz_model(V, source)

        J = assemble(1e6 * (u - u_ref)**2*dx)
        rf = ReducedFunctional(J, c)
        return rf

    assert len(get_working_tape()._blocks) == 0
    source = Function(V)
    rf = forward(source)
    with stop_annotating():
        solver = TAOSolver(
            MinimizationProblem(rf), tao_parameters,
            convert_options={"riesz_representation": riesz_representation})
        x = solver.solve()
        assert_allclose(x.dat.data, source_ref.dat.data, rtol=1e-2)

        get_working_tape().clear_tape()
        source_transform = transform(Function(V), TransformType.DUAL,
                                     riesz_representation,
                                     mfn_parameters=mfn_parameters)

    def forward_transform(source):
        c = Control(source)
        source = transform(source, TransformType.PRIMAL,
                           riesz_representation,
                           mfn_parameters=mfn_parameters)
        u = _simple_helmholz_model(V, source)

        J = assemble(1e6 * (u - u_ref)**2*dx)
        rf = ReducedFunctional(J, c)
        return rf
    rf_transform = forward_transform(source_transform)

    with stop_annotating():
        solver_transform = TAOSolver(
            MinimizationProblem(rf_transform), tao_parameters,
            convert_options={"riesz_representation": "l2"})
        x_transform = transform(solver_transform.solve(), TransformType.PRIMAL,
                                riesz_representation,
                                mfn_parameters=mfn_parameters)
        assert_allclose(x_transform.dat.data, source_ref.dat.data, rtol=1e-2)

        assert solver.tao.getIterationNumber() <= solver_transform.tao.getIterationNumber()


@pytest.mark.skipcomplex
def test_tao_bounds():
    mesh = UnitIntervalMesh(11)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    u = Function(space, name="u")
    u_ref = Function(space, name="u_ref").interpolate(0.5 - X[0])

    J = assemble((u - u_ref) ** 2 * dx)
    rf = ReducedFunctional(J, Control(u))

    lb = 0.5 - 7.0 / 11.0
    problem = MinimizationProblem(rf, bounds=(lb, None))
    solver = TAOSolver(problem, {"tao_type": "bnls",
                                 "tao_gatol": 1.0e-7,
                                 "tao_grtol": 0.0,
                                 "tao_gttol": 0.0})
    u_opt = solver.solve()

    u_ref_bound = u_ref.copy(deepcopy=True)
    u_ref_bound.dat.data[:] = np.maximum(u_ref_bound.dat.data_ro, lb)
    assert_allclose(u_opt.dat.data_ro, u_ref_bound.dat.data_ro, rtol=1.0e-2)
