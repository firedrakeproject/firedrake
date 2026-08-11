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


def _enriched_prefix(prefix: str) -> str:
    """Return the options prefix of the enriched-order primal solve.

    The enriched solve always keeps its own ``dwr_enriched_`` prefix. Its
    monitors and its unused options are then attributable to it, rather than
    mixed in with the parent's. When nothing is set under that prefix, the
    parent's options are copied across. By default the enriched problem is
    therefore solved the same way as the problem it enriches.

    Parameters
    ----------
    prefix
        The options prefix of the solver this callback is attached to.

    Returns
    -------
    The options prefix to give the enriched-order solver.
    """
    enriched_prefix = prefix + "dwr_enriched_"
    options = PETSc.Options(enriched_prefix)
    if not options.getAll():
        # These configure this callback and the adaptive loop driving it, not
        # the solve. Inheriting snes_adapt_sequence in particular would set the
        # enriched solve adapting a mesh of its own.
        not_inherited = ("dwr_", "snes_adapt_", "adaptor_")
        for key, value in PETSc.Options(prefix).getAll().items():
            if not key.startswith(not_inherited):
                options[key] = value
    return enriched_prefix


def _residual_indicators(F, dual_error, residual_degree, options_prefix):
    """Compute the strong residual representation of Rognes and Logg."""
    v, = F.arguments()
    V = v.function_space()
    mesh = V.mesh().unique()
    dim = mesh.topological_dimension
    degree = V.ufl_element().degree() + residual_degree
    variant = "integral"

    # Bubble functions vanish on cell boundaries, so testing F against
    # bubble * cell_test isolates the interior (strong) cell residual.
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
        cell_problem, options_prefix=options_prefix + "dwr_cell_"
    )
    cell_solver.solve()

    # Facet-bubble ("cone") functions vanish away from a single facet, so
    # subtracting off the already-localized cell_residual and testing
    # against facet_test isolates each facet's jump residual.
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
        facet_problem, options_prefix=options_prefix + "dwr_facet_"
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
        A scalar UFL 0-form depending on the primal solution.
    exact_solution
        An optional UFL expression for the exact primal solution. It is used
        only for diagnostics: it turns ``-dwr_monitor`` output into a true
        error and an effectivity index, and never influences marking.

    Notes
    -----
    Suitable for use as ``solve(..., marking_callback=DWRMarkingCallback(goal))``.
    Options are read from the options prefix of the solver this callback is
    attached to, and keep coming from there as the mesh is adapted. The
    supported options are ``dwr_enrichment_degree`` (default 1),
    ``dwr_residual_degree`` (default 1), ``dwr_marking_fraction``
    (default 0.5), ``dwr_atol`` (default 1e-50), ``dwr_rtol`` (default 0)
    and ``dwr_monitor`` (default off). The auxiliary solvers use the
    ``dwr_enriched_``, ``dwr_cell_``, and ``dwr_facet_`` sub-prefixes. The
    dual solves reuse the low- and enriched-order primal solvers' Jacobian
    via ``solve_jacobian_transpose``. If no ``dwr_enriched_`` options are
    set, the enriched-order solve inherits the parent solver's own options.

    Adaptation stops once ``|eta| < max(dwr_atol, dwr_rtol * |J(u_h)|)``,
    at which point the callback returns `None` rather than a set of markers.
    The default tolerances never trigger, so adaptation runs for the full
    ``-snes_adapt_sequence`` unless a tolerance is asked for.
    """

    def __init__(self, goal_functional: ufl.BaseForm,
                 exact_solution: ufl.classes.Expr | None = None,
                 primal: Function | None = None,
                 enrichment_degree: int | None = None,
                 options_prefix: str = ""):
        if not isinstance(goal_functional, ufl.BaseForm) or goal_functional.arguments():
            raise ValueError("goal_functional must be a 0-form")
        self.goal_functional = goal_functional
        self.exact_solution = exact_solution
        # The estimate of J(u) - J(u_h) from the most recent marking, and
        # whether it met the requested tolerances.
        self.error_estimate = None
        self.converged = False
        self._primal = primal
        self._enrichment_degree = enrichment_degree
        self._options_prefix = options_prefix
        self._high_space = None
        if primal is not None:
            V = primal.function_space()
            self._high_space = V.reconstruct(degree=V.ufl_element().degree() + enrichment_degree)

    def setup(self, primal: Function, options_prefix: str) -> None:
        options = PETSc.Options(options_prefix)
        self._options_prefix = options_prefix
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

        Weighting the residual by the dual error estimates the error committed
        by discretising. Weighting it by the dual itself estimates the error
        committed by not solving the algebraic system exactly. Their sum
        estimates ``J(u) - J(u_h)``.

        Parameters
        ----------
        problem
            The variational problem solved on the current mesh.
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
        # The options belong to the solver this callback was attached to.
        # Reconstructing the context onto a refined mesh renames its prefix
        # after the multigrid level it becomes. Nobody sets dwr_ options there.
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
        ctx.solve_jacobian_transpose(rhs, dual_low)

        primal_high = Function(high_space, name="dwr_primal_high")
        primal_high.interpolate(current_solution)
        high_problem = problem.rediscretise(u=primal_high)

        nullspace = None if ctx._nullspace is None else ctx._nullspace.rediscretise(high_space)
        transpose_nullspace = None if ctx._nullspace_T is None else ctx._nullspace_T.rediscretise(high_space)
        near_nullspace = None if ctx._near_nullspace is None else ctx._near_nullspace.rediscretise(high_space)
        primal_solver = NonlinearVariationalSolver(
            high_problem,
            options_prefix=_enriched_prefix(prefix),
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace,
        )
        primal_solver.solve()

        goal_high = replace(self.goal_functional, {current_solution: primal_high})
        dual_high = Function(high_space, name="dwr_dual_high")
        dual_test_high = TestFunction(high_space)
        goal_derivative_high = derivative(goal_high, primal_high, dual_test_high)
        rhs_high = assemble(goal_derivative_high, bcs=high_problem.bcs)
        primal_solver._ctx.solve_jacobian_transpose(rhs_high, dual_high)

        dual_error = dual_high - dual_low
        self.error_estimate = self._estimate_error(
            problem, current_solution, dual_low, dual_error, prefix
        )
        if self.converged:
            # Localizing the estimate onto cells costs two more solves, and
            # nothing is left to refine.
            return None

        indicators = _residual_indicators(
            problem.F, dual_error, residual_degree, prefix
        )
        return _dorfler_mark(indicators, marking_fraction)
