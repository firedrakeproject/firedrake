from __future__ import annotations

import numpy as np
import ufl
from finat.ufl import BrokenElement, FiniteElement
from petsctools.options import DefaultOptionSet, get_default_options

from firedrake.assemble import assemble
from firedrake.exceptions import ConvergenceError
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.logging import RED, warning
from firedrake.petsc import PETSc
from firedrake.ufl_expr import TestFunction, TrialFunction, derivative
from firedrake.variational_solver import (LinearVariationalProblem,
                                          LinearVariationalSolver,
                                          NonlinearVariationalSolver)
from ufl import avg, dS, ds, dx, inner, replace


__all__ = ("DWRMarkingCallback",)


def _replace_arguments(form, *arguments):
    return replace(form, dict(zip(form.arguments(), arguments)))


def _both(expr):
    return expr("+") + expr("-")


def _residual_indicators(F, dual_error, residual_degree, options_prefix):
    """Compute the strong residual representation of Rognes and Logg.

    Parameters
    ----------
    F
        The residual form of the primal problem.
    dual_error
        The dual error representative ``z - z_h``.
    residual_degree
        The number of degrees that the localization spaces add to the primal
        space.
    options_prefix
        The options prefix of the solver that this callback is attached to.

    Returns
    -------
    A DG0 `~firedrake.function.Function` that holds one error indicator per
    cell.
    """
    v, = F.arguments()
    V = v.function_space()
    mesh = V.mesh().unique()
    dim = mesh.topological_dimension
    degree = V.ufl_element().degree() + residual_degree
    variant = "integral"

    # Testing F against cell bubbles isolates the cell residual.
    bubble_space = FunctionSpace(mesh, "B", dim + 1, variant=variant)
    bubble = Function(bubble_space).assign(1)
    if V.value_shape == ():
        cell_space = FunctionSpace(mesh, "DG", degree, variant=variant)
    else:
        cell_space = TensorFunctionSpace(mesh, "DG", degree,
                                         shape=V.value_shape, variant=variant)
    cell_trial = TrialFunction(cell_space)
    cell_test = TestFunction(cell_space)
    cell_residual = Function(cell_space)
    cell_problem = LinearVariationalProblem(
        inner(cell_trial, bubble * cell_test) * dx,
        _replace_arguments(F, bubble * cell_test), cell_residual,
    )
    cell_solver = LinearVariationalSolver(
        cell_problem, options_prefix=options_prefix + "dwr_cell_",
    )
    cell_solver.solve()

    # Testing the remainder against facet bubbles isolates the facet residual.
    cone_space = FunctionSpace(mesh, "FB", dim, variant=variant)
    cone = Function(cone_space).assign(1)
    element = BrokenElement(FiniteElement("FB", cell=mesh.ufl_cell(),
                                          degree=degree + dim, variant=variant))
    if V.value_shape == ():
        facet_space = FunctionSpace(mesh, element)
    else:
        facet_space = TensorFunctionSpace(mesh, element, shape=V.value_shape)
    facet_trial = TrialFunction(facet_space)
    facet_test = TestFunction(facet_space)
    facet_residual_hat = Function(facet_space)
    facet_rhs = (_replace_arguments(F, facet_test)
                 - inner(cell_residual, facet_test) * dx)
    facet_lhs = (_both(inner(facet_trial / cone, facet_test)) * dS
                 + inner(facet_trial / cone, facet_test) * ds)
    facet_problem = LinearVariationalProblem(
        facet_lhs, facet_rhs, facet_residual_hat
    )
    facet_solver = LinearVariationalSolver(
        facet_problem, options_prefix=options_prefix + "dwr_facet_",
    )
    facet_solver.solve()
    facet_residual = facet_residual_hat / cone

    indicator_space = FunctionSpace(mesh, "DG", 0)
    indicator_test = TestFunction(indicator_space)
    indicators = assemble(
        inner(inner(cell_residual, dual_error), indicator_test) * dx
        + inner(avg(inner(facet_residual, dual_error)),
                _both(indicator_test)) * dS
        + inner(inner(facet_residual, dual_error), indicator_test) * ds
    )
    with indicators.dat.vec as vec:
        vec.abs()
    return indicators


def _dorfler_mark(indicators: Function, fraction: float) -> Function:
    if not 0 < fraction <= 1:
        raise ValueError("marking_fraction must lie in (0, 1]")
    local = indicators.dat.data_ro.copy()
    if not np.isfinite(local).all():
        raise ConvergenceError("DWR error indicators contain non-finite values")
    gathered = indicators.comm.allgather(local)
    values = np.concatenate(gathered)
    total = values.sum()
    markers = Function(indicators.function_space())
    if total <= 0:
        return markers.assign(1)
    ordered = np.sort(values)[::-1]
    count = np.searchsorted(np.cumsum(ordered), fraction * total) + 1
    threshold = ordered[min(count - 1, len(ordered) - 1)]
    markers.dat.data_wo[:] = local >= threshold
    return markers


class DWRMarkingCallback:
    """Mark cells using an automatically localized dual-weighted residual.

    Parameters
    ----------
    goal_functional
        A scalar UFL 0-form that depends on the primal solution.
    exact_solution
        An optional UFL expression for the exact primal solution.
        ``-dwr_monitor`` uses it to report the true error and the effectivity
        index.

    Attributes
    ----------
    goal_functional
        The goal functional on the current mesh.
    error_estimate
        The most recent estimate ``eta`` of ``J(u) - J(u_h)``, or `None`
        before the first marking. It belongs to the current mesh if
        adaptation stopped at the tolerance, and to the previous mesh
        otherwise.
    converged
        Whether ``error_estimate`` meets the tolerances.

    Notes
    -----
    Use as ``solve(..., marking_callback=DWRMarkingCallback(goal))``.
    The callback reads these options from the prefix of its solver:
    ``dwr_enrichment_degree`` (default 1), ``dwr_residual_degree``
    (default 1), ``dwr_marking_fraction`` (default 0.5), ``dwr_atol``
    (default 1e-50), ``dwr_rtol`` (default 0) and ``dwr_monitor``
    (default off). The auxiliary solvers read the ``dwr_enriched_``,
    ``dwr_cell_`` and ``dwr_facet_`` sub-prefixes. The enriched solve
    inherits the options of the parent solver, except the ``dwr_`` options
    and ``snes_adapt_sequence``, and its ``dwr_enriched_`` options take
    precedence. `NonlinearVariationalSolver.get_marking_callback` returns the
    callback on the current mesh. The dual solves reuse the primal Jacobians
    through ``solve_jacobian``, so the preconditioners of both primal solvers
    must implement ``applyTranspose``.

    Adaptation stops once ``|eta| < max(dwr_atol, dwr_rtol * |J(u_h)|)``, and
    the callback then returns `None`. With the default tolerances, adaptation
    runs for the whole ``-snes_adapt_sequence``.
    """

    def __init__(self, goal_functional: ufl.BaseForm,
                 exact_solution: ufl.classes.Expr | None = None):
        if not isinstance(goal_functional, ufl.BaseForm) or goal_functional.arguments():
            raise ValueError("goal_functional must be a 0-form")
        self.goal_functional = goal_functional
        self.exact_solution = exact_solution
        self.error_estimate = None
        self.converged = False
        # The prefix of the solver that this callback is attached to.
        self._options_prefix = None

    def reconstruct(self, goal_functional: ufl.BaseForm,
                    exact_solution: ufl.classes.Expr | None) -> DWRMarkingCallback:
        """Return a copy of this callback for another mesh.

        Parameters
        ----------
        goal_functional
            The goal functional on the other mesh.
        exact_solution
            The exact solution on the other mesh, or `None`.

        Returns
        -------
        A callback that keeps the options prefix and the error estimate of this one.
        """
        callback = type(self)(goal_functional, exact_solution)
        callback.error_estimate = self.error_estimate
        callback._options_prefix = self._options_prefix
        return callback

    def __call__(self, ctx, current_solution: Function) -> Function | None:
        return self._mark(ctx, current_solution)

    def _estimate_error(self, problem, current_solution: Function,
                        dual_low: Function, dual_error: ufl.classes.Expr,
                        options_prefix: str) -> float:
        """Estimate ``J(u) - J(u_h)``, and report on it.

        The estimate is the sum of the discretisation error
        ``rho(u_h; z - z_h)`` and the solver error ``rho(u_h; z_h)``.

        Parameters
        ----------
        problem
            The variational problem on the current mesh.
        current_solution
            The primal solution ``u_h``.
        dual_low
            The dual solution ``z_h`` in the primal space.
        dual_error
            The dual error representative ``z - z_h``.
        options_prefix
            The PETSc options prefix of the active solver.

        Returns
        -------
        The error estimate ``eta``.
        """
        options = PETSc.Options(options_prefix)
        discretisation_error = assemble(_replace_arguments(problem.F, -dual_error))
        solver_error = assemble(_replace_arguments(problem.F, -dual_low))
        error_estimate = discretisation_error + solver_error
        goal = assemble(self.goal_functional)

        atol = options.getReal("dwr_atol", 1e-50)
        rtol = options.getReal("dwr_rtol", 0.0)
        self.converged = abs(error_estimate) < max(atol, rtol * abs(goal))

        if abs(solver_error) > abs(discretisation_error):
            warning(RED % ("DWR: the solver error exceeds the discretisation error. "
                           "Tighten the solver tolerances."))

        if options.getBool("dwr_monitor", False):
            report = [("goal J(u_h)", goal),
                      ("discretisation error rho(u_h; z-z_h)", discretisation_error),
                      ("solver error rho(u_h; z_h)", solver_error),
                      ("error estimate eta", error_estimate)]
            if self.exact_solution is not None:
                exact_goal = assemble(replace(self.goal_functional,
                                              {current_solution: self.exact_solution}))
                true_error = exact_goal - goal
                report.append(("exact goal J(u)", exact_goal))
                report.append(("true error J(u) - J(u_h)", true_error))
                if true_error != 0:
                    report.append(("effectivity index", error_estimate / true_error))
            for label, value in report:
                PETSc.Sys.Print(f"    DWR {label:<38s}{value: 15.8e}",
                                comm=current_solution.comm)
        return error_estimate

    def _mark(self, ctx, current_solution: Function) -> Function | None:
        problem = ctx._problem
        V = current_solution.function_space()
        if self._options_prefix is None:
            self._options_prefix = ctx.options_prefix or ""
        # Refined contexts have a prefix for their multigrid level, so read the
        # options of the original solver.
        prefix = self._options_prefix
        options = PETSc.Options(prefix)
        enrichment_degree = options.getInt("dwr_enrichment_degree", 1)
        residual_degree = options.getInt("dwr_residual_degree", 1)
        marking_fraction = options.getReal("dwr_marking_fraction", 0.5)
        high_space = V.reconstruct(degree=V.ufl_element().degree() + enrichment_degree)

        dual_low = Function(V, name="dwr_dual_low")
        goal_derivative = derivative(self.goal_functional, current_solution)
        rhs = assemble(goal_derivative, bcs=problem.bcs)
        ctx.solve_jacobian(rhs, dual_low, transpose=True)

        primal_high = Function(high_space, name="dwr_primal_high")
        primal_high.interpolate(current_solution)
        high_problem = problem.rediscretise(u=primal_high)

        nullspace = None if ctx._nullspace is None else ctx._nullspace.rediscretise(high_space)
        transpose_nullspace = None if ctx._nullspace_T is None else ctx._nullspace_T.rediscretise(high_space)
        near_nullspace = None if ctx._near_nullspace is None else ctx._near_nullspace.rediscretise(high_space)
        parameters = get_default_options(DefaultOptionSet(prefix, ("dwr_",)))
        parameters.pop("snes_adapt_sequence", None)
        primal_solver = NonlinearVariationalSolver(
            high_problem,
            options_prefix=prefix + "dwr_enriched_",
            solver_parameters=parameters,
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace,
        )
        primal_solver.solve()

        goal_high = replace(self.goal_functional, {current_solution: primal_high})
        dual_high = Function(high_space, name="dwr_dual_high")
        goal_derivative_high = derivative(goal_high, primal_high)
        rhs_high = assemble(goal_derivative_high, bcs=high_problem.bcs)
        primal_solver._ctx.solve_jacobian(rhs_high, dual_high, transpose=True)

        dual_error = dual_high - dual_low
        self.error_estimate = self._estimate_error(
            problem, current_solution, dual_low, dual_error, prefix
        )
        if self.converged:
            return None

        indicators = _residual_indicators(
            problem.F, dual_error, residual_degree, prefix
        )
        return _dorfler_mark(indicators, marking_fraction)
