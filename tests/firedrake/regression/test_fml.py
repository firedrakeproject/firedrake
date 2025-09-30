"""
Tests a full workflow for the Form Manipulation Language (FML).

This uses an IMEX discretisation of the linear shallow-water equations on a
mixed function space.
"""

from firedrake import (
    PeriodicUnitSquareMesh, FunctionSpace, Constant,
    MixedFunctionSpace, TestFunctions, Function, split, inner, dx,
    SpatialCoordinate, as_vector, pi, sin, div,
    NonlinearVariationalProblem, NonlinearVariationalSolver
)
from firedrake.fml import subject, replace_subject, keep, drop, Label


def test_fml():

    # Define labels for shallow-water
    time_derivative = Label("time_derivative")
    transport = Label("transport")
    pressure_gradient = Label("pressure_gradient")
    explicit = Label("explicit")
    implicit = Label("implicit")

    # ------------------------------------------------------------------------ #
    # Set up finite element objects
    # ------------------------------------------------------------------------ #

    # Two shallow-water constants
    H = Constant(10000.)
    g = Constant(10.)

    # Set up mesh and function spaces
    dt = Constant(0.01)
    Nx = 5
    mesh = PeriodicUnitSquareMesh(Nx, Nx)
    spaces = [FunctionSpace(mesh, "BDM", 1), FunctionSpace(mesh, "DG", 1)]
    W = MixedFunctionSpace(spaces)

    # Set up fields on a mixed function space
    w, phi = TestFunctions(W)
    X = Function(W)
    u0, h0 = split(X)

    # Set up time derivatives
    mass_form = time_derivative(subject(inner(u0, w)*dx + subject(inner(h0, phi)*dx), X))

    # Height field transport form
    transport_form = transport(subject(H*inner(div(u0), phi)*dx, X))

    # Pressure gradient term -- integrate by parts once
    pressure_gradient_form = pressure_gradient(subject(-g*inner(h0, div(w))*dx, X))

    # Define IMEX scheme. Transport term explicit and pressure gradient implict.
    # This is not necessarily a sensible scheme -- it's just a simple demo for
    # how FML can be used.
    transport_form = explicit(transport_form)
    pressure_gradient_form = implicit(pressure_gradient_form)

    # Add terms together to give whole residual
    residual = mass_form + transport_form + pressure_gradient_form

    # ------------------------------------------------------------------------ #
    # Initial condition
    # ------------------------------------------------------------------------ #

    # Constant flow but sinusoidal height field
    x, _ = SpatialCoordinate(mesh)
    u0, h0 = X.subfunctions
    u0.interpolate(as_vector([1.0, 0.0]))
    h0.interpolate(H + 0.01*H*sin(2*pi*x))

    # ------------------------------------------------------------------------ #
    # Set up time discretisation
    # ------------------------------------------------------------------------ #

    X_np1 = Function(W)

    # Here we would normally set up routines for the explicit and implicit parts
    # but as this is just a test, we'll do just a single explicit/implicit step

    # Explicit: just forward euler
    explicit_lhs = residual.label_map(lambda t: t.has_label(time_derivative),
                                      map_if_true=replace_subject(X_np1),
                                      map_if_false=drop)

    explicit_rhs = residual.label_map(lambda t: t.has_label(time_derivative)
                                      or t.has_label(explicit),
                                      map_if_true=keep, map_if_false=drop)
    explicit_rhs = explicit_rhs.label_map(lambda t: t.has_label(time_derivative),
                                          map_if_false=lambda t: -dt*t)

    # Implicit: just backward euler
    implicit_lhs = residual.label_map(lambda t: t.has_label(time_derivative)
                                      or t.has_label(implicit),
                                      map_if_true=replace_subject(X_np1),
                                      map_if_false=drop)
    implicit_lhs = implicit_lhs.label_map(lambda t: t.has_label(time_derivative),
                                          map_if_false=lambda t: dt*t)

    implicit_rhs = residual.label_map(lambda t: t.has_label(time_derivative),
                                      map_if_false=drop)

    # ------------------------------------------------------------------------ #
    # Set up and solve problems
    # ------------------------------------------------------------------------ #

    explicit_residual = explicit_lhs - explicit_rhs
    implicit_residual = implicit_lhs - implicit_rhs

    explicit_problem = NonlinearVariationalProblem(explicit_residual.form, X_np1)
    explicit_solver = NonlinearVariationalSolver(explicit_problem)

    implicit_problem = NonlinearVariationalProblem(implicit_residual.form, X_np1)
    implicit_solver = NonlinearVariationalSolver(implicit_problem)

    # Solve problems and update X_np1
    # In reality this would be within a time stepping loop!
    explicit_solver.solve()
    X.assign(X_np1)
    implicit_solver.solve()
    X.assign(X_np1)
