import pytest
from firedrake import *
from firedrake.adjoint import *
from firedrake.bcs import DirichletBC
from firedrake.rf_eigensolver import *
from firedrake.adjoint import PETScVecInterface
from firedrake.restricted_functional_ctx import (
    new_restricted_control_variable,
    interpolate_vars,
)


def generate_expressions():
    """
    Returns a list of three expressions.
    """
    mesh = UnitSquareMesh(50, 50, quadrilateral=False)
    x, y = SpatialCoordinate(mesh)
    return [2 * x, exp(-x - y), sin(pi * x * y)]


@pytest.mark.parametrize("expression", generate_expressions())
def test_compare_eigensolvers_helmholtz(expression):
    """Test TLM vs assembled form for Helmholtz eigenproblem."""
    n = 50
    num_eigenvalues = 10

    mesh = UnitSquareMesh(n, n, quadrilateral=False)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0.0, "on_boundary")

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(grad(u), grad(v)) + u * v) * dx

    # Form based Ax = cMx
    problem_form = LinearEigenproblem(a, bcs=bc, restrict=True)
    solver_form = LinearEigensolver(problem_form, n_evals=num_eigenvalues)
    solver_form.solve()
    nconv = solver_form.es.getConverged()

    evals_form = [
        solver_form.eigenvalue(i) for i in range(0, min(num_eigenvalues, nconv))
    ]

    # ReducedFunctional-based, Jx = 1/cx
    continue_annotation()
    f = Function(V)
    f.interpolate(2 * x)
    control = Control(f)
    u_1 = Function(V)
    v = TestFunction(V)
    F = (inner(grad(u_1), grad(v)) + u_1 * v - f * v) * dx
    solve(F == 0, u_1, bcs=bc)
    J = ReducedFunctional(u_1, controls=[control])
    pause_annotation()

    problem_rf = RFEigenproblem(J, bcs=bc, apply_riesz=False, restrict=True, identity=True)
    solver_rf = RFEigensolver(problem_rf, n_evals=num_eigenvalues)
    solver_rf.es.setTarget(1)
    solver_rf.solve()
    nconv = solver_rf.es.getConverged()

    evals_rf = [
        1 / solver_rf.eigenvalue(i) for i in range(0, min(num_eigenvalues, nconv))
    ]

    for ev1, ev2 in zip(evals_form, evals_rf):
        assert abs(ev1 - ev2) < 1e-6

    set_working_tape(Tape())


@pytest.mark.parametrize("restrict", [False])
@pytest.mark.parametrize("riesz", ["L2"])
def test_apply_riesz_param(restrict, riesz):
    """Test implicit matrix for Helmholtz eigenproblem."""
    n = 50
    mesh = UnitSquareMesh(n, n, quadrilateral=False)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0.0, "on_boundary")

    # Tape the solve
    continue_annotation()
    f = Function(V)
    f.interpolate(2 * x * y)
    control = Control(f)
    u_1 = Function(V)
    v = TestFunction(V)
    F = (inner(grad(u_1), grad(v)) + u_1 * v - f * v) * dx
    solve(F == 0, u_1, bcs=bc)
    J = ReducedFunctional(u_1, controls=[control])
    pause_annotation()

    # Check adjoint action
    problem_rf_true = RFEigenproblem(
        J, bcs=bc, apply_riesz=True, restrict=restrict, action="adjoint"
    )
    problem_rf_false = RFEigenproblem(
        J, bcs=bc, apply_riesz=False, restrict=restrict, action="adjoint"
    )

    mat_true = problem_rf_true.A
    mat_false = problem_rf_false.A

    x = mat_true.createVecRight()
    Ax = mat_true.createVecLeft()
    Ax2 = mat_true.createVecLeft()
    x.setRandom()

    mat_true.mult(x, Ax)
    mat_false.mult(x, Ax2)

    interface_space = problem_rf_true.restricted_space if restrict else V

    interface_Ax = PETScVecInterface(
        [Function(interface_space).interpolate(c.control) for c in J.controls]
    )
    func_representation = new_restricted_control_variable(J, interface_space, dual=False)
    func_representation_dual = new_restricted_control_variable(
        J, interface_space, dual=True
    )
    interface_Ax.from_petsc(Ax, func_representation)
    interface_Ax.from_petsc(Ax2, func_representation_dual)

    if restrict:
        # Interpolate from restricted to unrestricted
        func_representation = interpolate_vars(func_representation, V)
        func_representation_dual = interpolate_vars(func_representation_dual, V)

    set_working_tape(Tape())
    assert (
        errornorm(
            func_representation[0],
            func_representation_dual[0].riesz_representation(riesz_map=riesz),
        )
        < 1e-6
    )


def test_compare_tlm_adjoint():
    """Test TLM and adjoint give same eigenvalues for Helmholtz eigenproblem."""
    n = 50
    num_eigenvalues = 10

    mesh = UnitSquareMesh(n, n, quadrilateral=False)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0.0, "on_boundary")

    # ReducedFunctional-based, Jx = 1/cx
    continue_annotation()
    f = Function(V)
    f.interpolate(2 * x)
    control = Control(f)
    u_1 = Function(V)
    v = TestFunction(V)
    F = (inner(grad(u_1), grad(v)) + u_1 * v - f * v) * dx
    solve(F == 0, u_1, bcs=bc)
    J = ReducedFunctional(u_1, controls=[control])
    pause_annotation()

    problem_rf_a = RFEigenproblem(
        J, bcs=bc, apply_riesz=False, restrict=True, identity=True, action="adjoint"
    )
    solver_rf_a = RFEigensolver(problem_rf_a, n_evals=num_eigenvalues)
    solver_rf_a.es.setTarget(1)
    solver_rf_a.solve()
    nconv = solver_rf_a.es.getConverged()

    evals_rf_a = [
        solver_rf_a.eigenvalue(i) for i in range(0, min(num_eigenvalues, nconv))
    ]

    problem_rf_t = RFEigenproblem(
        J, bcs=bc, apply_riesz=False, restrict=True, identity=True, action="tlm"
    )
    solver_rf_t = RFEigensolver(problem_rf_t, n_evals=num_eigenvalues)
    solver_rf_t.es.setTarget(1)
    solver_rf_t.solve()
    nconv = solver_rf_t.es.getConverged()

    evals_rf_t = [
        solver_rf_t.eigenvalue(i) for i in range(0, min(num_eigenvalues, nconv))
    ]
    for ev1, ev2 in zip(evals_rf_a, evals_rf_t):
        print(abs(ev1 - ev2))
        assert abs(ev1 - ev2) < 1e-6

    set_working_tape(Tape())
