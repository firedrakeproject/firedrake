from __future__ import annotations

import numpy as np
import ufl
from finat.ufl import BrokenElement, FiniteElement

from firedrake.assemble import assemble
from firedrake.exceptions import ConvergenceError
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.petsc import PETSc
from firedrake.ufl_expr import TestFunction, TrialFunction, derivative
from firedrake.variational_solver import (LinearVariationalProblem,
                                          LinearVariationalSolver,
                                          NonlinearVariationalSolver)
from ufl import avg, dS, ds, dx, inner, replace


__all__ = ("DWRMarkingCallback",)


def _replace_arguments(form, *arguments):
    return replace(form, dict(zip(form.arguments(), arguments)))


def _homogeneous_bcs(bcs, V):
    return [bc.reconstruct(V=V, indices=bc._indices, g=0) for bc in bcs]


def _both(expr):
    return expr("+") + expr("-")


def _enriched_prefix(prefix):
    """Use the ``dwr_enriched_`` sub-prefix only if it is configured,
    otherwise inherit the parent solver's options."""
    enriched_prefix = prefix + "dwr_enriched_"
    if PETSc.Options(enriched_prefix).getAll():
        return enriched_prefix
    return prefix


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

    Notes
    -----
    Suitable for use as ``solve(..., marking_callback=DWRMarkingCallback(goal))``.
    Options are read from the active solver's PETSc options prefix. The
    supported options are ``dwr_enrichment_degree`` (default 1),
    ``dwr_residual_degree`` (default 1), and ``dwr_marking_fraction``
    (default 0.5). The auxiliary solvers use the ``dwr_enriched_``,
    ``dwr_cell_``, and ``dwr_facet_`` sub-prefixes. The dual solves reuse
    the low- and enriched-order primal solvers' Jacobian via
    ``solve_jacobian_transpose``. If no ``dwr_enriched_`` options are set,
    the enriched-order solve inherits the parent solver's own options.
    """

    def __init__(self, goal_functional: ufl.BaseForm,
                 primal: Function | None = None,
                 enrichment_degree: int | None = None):
        if not isinstance(goal_functional, ufl.BaseForm) or goal_functional.arguments():
            raise ValueError("goal_functional must be a 0-form")
        self.goal_functional = goal_functional
        self._primal = primal
        self._enrichment_degree = enrichment_degree
        self._high_space = None
        if primal is not None:
            V = primal.function_space()
            self._high_space = V.reconstruct(degree=V.ufl_element().degree() + enrichment_degree)

    def setup(self, primal: Function, options_prefix: str) -> None:
        options = PETSc.Options(options_prefix)
        self._primal = primal
        self._enrichment_degree = options.getInt("dwr_enrichment_degree", 1)
        V = primal.function_space()
        self._high_space = V.reconstruct(degree=V.ufl_element().degree() + self._enrichment_degree)

    def __call__(self, ctx, current_solution: Function) -> Function:
        return self._mark(ctx, current_solution)

    def _mark(self, ctx, current_solution: Function) -> Function:
        problem = ctx._problem
        V = current_solution.function_space()
        prefix = ctx.options_prefix or ""
        options = PETSc.Options(prefix)
        residual_degree = options.getInt("dwr_residual_degree", 1)
        marking_fraction = options.getReal("dwr_marking_fraction", 0.5)
        high_space = self._high_space
        if high_space is None:
            raise RuntimeError("DWR marking callback has not been set up")

        dual_low = Function(V, name="dwr_dual_low")
        direction = TestFunction(V)
        goal_derivative = derivative(self.goal_functional, current_solution, direction)
        rhs = assemble(goal_derivative, bcs=_homogeneous_bcs(problem.bcs, V))
        ctx.solve_jacobian_transpose(rhs, dual_low)

        primal_high = Function(high_space, "dwr_primal_high")
        primal_high.interpolate(current_solution)
        high_problem = problem.reconstruct(u=primal_high)

        nullspace = None if ctx._nullspace is None else ctx._nullspace.reconstruct(high_space)
        transpose_nullspace = None if ctx._nullspace_T is None else ctx._nullspace_T.reconstruct(high_space)
        near_nullspace = None if ctx._near_nullspace is None else ctx._near_nullspace.reconstruct(high_space)
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
        indicators = _residual_indicators(
            problem.F, dual_error, residual_degree, prefix
        )
        return _dorfler_mark(indicators, marking_fraction)
