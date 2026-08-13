from __future__ import annotations

import numpy as np
import ufl
from finat.ufl import BrokenElement, FiniteElement

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


def _sub_parameters(options: dict[str, str], sub_prefix: str) -> dict[str, str]:
    """Return the options of one auxiliary solver.

    Parameters
    ----------
    options
        The options under the prefix of the parent solver.
    sub_prefix
        The sub-prefix of one auxiliary solver, such as ``"dwr_cell_"``.

    Returns
    -------
    The options that begin with ``sub_prefix``, with that sub-prefix removed.
    """
    return {key[len(sub_prefix):]: value
            for key, value in options.items() if key.startswith(sub_prefix)}


def _enriched_parameters(options: dict[str, str]) -> dict[str, str]:
    """Return the solver parameters of the enriched-order primal solve.

    The enriched solve keeps its own ``dwr_enriched_`` prefix. Its monitors and
    its unused options then belong to it alone, and do not mix with those of
    the parent. If the user sets no option under that prefix, this function
    copies the parent's options across. The enriched problem then uses the same
    solver as the problem that it enriches.

    Parameters
    ----------
    options
        The options under the prefix of the parent solver.

    Returns
    -------
    The solver parameters to give the enriched-order solver.
    """
    parameters = _sub_parameters(options, "dwr_enriched_")
    if not parameters:
        # These options configure this callback and the adaptive loop that
        # drives it, not the solve itself. An enriched solve that inherited
        # snes_adapt_sequence would adapt a mesh of its own.
        not_inherited = ("dwr_", "snes_adapt_", "adaptor_")
        parameters = {key: value for key, value in options.items()
                      if not key.startswith(not_inherited)}
    return parameters


def _residual_indicators(F, dual_error, residual_degree, options_prefix, options):
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
    options
        The options under ``options_prefix``.

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

    # Bubble functions vanish on cell boundaries. A test of F against
    # bubble * cell_test therefore isolates the interior (strong) cell
    # residual.
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
        solver_parameters=_sub_parameters(options, "dwr_cell_"),
    )
    cell_solver.solve()

    # Facet-bubble ("cone") functions vanish away from a single facet. This
    # problem subtracts the cell residual that the first solve localized, and
    # tests the remainder against facet_test. That isolates the jump residual
    # of each facet.
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
        solver_parameters=_sub_parameters(options, "dwr_facet_"),
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
        An optional UFL expression for the exact primal solution. It serves
        diagnostics only: it turns ``-dwr_monitor`` output into a true error
        and an effectivity index, and never influences marking.
    primal
        The primal solution ``u_h``. `setup` sets it.
    enrichment_degree
        The number of degrees that the enriched space adds to the primal
        space. `setup` reads it from the options.
    options_prefix
        The options prefix of the solver that this callback is attached to.
        `setup` sets it.
    options
        The options under ``options_prefix``. `setup` captures them.

    Notes
    -----
    Suitable for use as ``solve(..., marking_callback=DWRMarkingCallback(goal))``.
    This callback reads its options from the prefix of the solver that it is
    attached to, and keeps reading them from there as the mesh adapts. The
    supported options are ``dwr_enrichment_degree`` (default 1),
    ``dwr_residual_degree`` (default 1), ``dwr_marking_fraction``
    (default 0.5), ``dwr_atol`` (default 1e-50), ``dwr_rtol`` (default 0)
    and ``dwr_monitor`` (default off). The auxiliary solvers use the
    ``dwr_enriched_``, ``dwr_cell_``, and ``dwr_facet_`` sub-prefixes. They
    keep the options that those sub-prefixes held when a solver attached this
    callback. If the user sets no ``dwr_enriched_`` option, the enriched-order
    solve inherits the parent solver's own options. The dual solves reuse the
    Jacobian of the low- and enriched-order primal solvers through
    ``solve_jacobian``, so a preconditioner that implements ``applyTranspose``
    must precondition both primal solvers.

    Adaptation stops once ``|eta| < max(dwr_atol, dwr_rtol * |J(u_h)|)``. The
    callback then returns `None` rather than a set of markers. The default
    tolerances never trigger, so adaptation runs for the whole
    ``-snes_adapt_sequence`` unless the user asks for a tolerance.
    """

    def __init__(self, goal_functional: ufl.BaseForm,
                 exact_solution: ufl.classes.Expr | None = None,
                 primal: Function | None = None,
                 enrichment_degree: int | None = None,
                 options_prefix: str = "",
                 options: dict[str, str] | None = None):
        if not isinstance(goal_functional, ufl.BaseForm) or goal_functional.arguments():
            raise ValueError("goal_functional must be a 0-form")
        self.goal_functional = goal_functional
        self.exact_solution = exact_solution
        # The estimate of J(u) - J(u_h) that the most recent marking gave, and
        # whether that estimate met the requested tolerances.
        self.error_estimate = None
        self.converged = False
        self._primal = primal
        self._enrichment_degree = enrichment_degree
        self._options_prefix = options_prefix
        # Each auxiliary solver deletes the options that it reads from the
        # database. This copy survives that deletion, and configures the
        # solvers that the callback rebuilds on every adapted mesh.
        self._options = {} if options is None else options
        self._high_space = None
        if primal is not None:
            V = primal.function_space()
            self._high_space = V.reconstruct(degree=V.ufl_element().degree() + enrichment_degree)

    def setup(self, primal: Function, options_prefix: str) -> None:
        """Attach this callback to a solver, and read that solver's options.

        Parameters
        ----------
        primal
            The primal solution ``u_h`` on the initial mesh.
        options_prefix
            The options prefix of that solver.
        """
        options = PETSc.Options(options_prefix)
        self._options_prefix = options_prefix
        self._options = options.getAll()
        self._primal = primal
        self._enrichment_degree = options.getInt("dwr_enrichment_degree", 1)
        V = primal.function_space()
        self._high_space = V.reconstruct(degree=V.ufl_element().degree() + self._enrichment_degree)

    def __call__(self, ctx, current_solution: Function) -> Function | None:
        return self._mark(ctx, current_solution)

    def _estimate_error(self, problem, current_solution: Function,
                        dual_low: Function, dual_error: ufl.classes.Expr,
                        options_prefix: str) -> float:
        """Estimate the error in the goal functional, and report on it.

        The residual, weighted by the dual error, estimates the error that the
        discretisation makes. The residual, weighted by the dual itself,
        estimates the error that the inexact algebraic solve makes. Their sum
        estimates ``J(u) - J(u_h)``.

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
            warning(RED % ("DWR: the solver error estimate exceeds the discretisation "
                           "error estimate, so the estimate is dominated by how "
                           "loosely the algebraic system was solved. Tighten the "
                           "solver tolerances."))

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
        # The options belong to the solver that this callback was attached to.
        # A context that moves onto a refined mesh takes its name from the
        # multigrid level that it becomes. Nobody sets dwr_ options there.
        prefix = self._options_prefix
        options = PETSc.Options(prefix)
        residual_degree = options.getInt("dwr_residual_degree", 1)
        marking_fraction = options.getReal("dwr_marking_fraction", 0.5)
        high_space = self._high_space
        if high_space is None:
            raise RuntimeError("DWR marking callback has not been set up")

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
        primal_solver = NonlinearVariationalSolver(
            high_problem,
            options_prefix=prefix + "dwr_enriched_",
            solver_parameters=_enriched_parameters(self._options),
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
            # Two more solves localize the estimate onto the cells, and no cell
            # remains to refine.
            return None

        indicators = _residual_indicators(
            problem.F, dual_error, residual_degree, prefix, self._options
        )
        return _dorfler_mark(indicators, marking_fraction)
