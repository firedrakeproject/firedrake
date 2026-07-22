import numpy as np
import ufl
from finat.ufl import BrokenElement, FiniteElement

from firedrake.assemble import assemble
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.petsc import PETSc
from firedrake.preconditioners.pmg import PMGPC
from firedrake.ufl_expr import TestFunction, TrialFunction, adjoint, derivative
from firedrake.variational_solver import (LinearVariationalProblem,
                                          LinearVariationalSolver,
                                          NonlinearVariationalProblem,
                                          NonlinearVariationalSolver)
from ufl import avg, dS, ds, dx, inner, replace


__all__ = ("dwr_marking_callback",)


def _replace_arguments(form, *arguments):
    return replace(form, dict(zip(form.arguments(), arguments)))


def _reconstruct_degree(V, degree):
    element = PMGPC.reconstruct_degree(V.ufl_element(), degree)
    return V.reconstruct(element=element)


def _homogeneous_bcs(bcs, V):
    return [bc.reconstruct(V=V, indices=bc._indices, g=0) for bc in bcs]


def _both(expr):
    return expr("+") + expr("-")


def _residual_indicators(F, dual_error, residual_degree, options_prefix):
    """Compute the strong residual representation of Rognes and Logg."""
    v, = F.arguments()
    V = v.function_space()
    mesh = V.mesh().unique()
    dim = mesh.topological_dimension
    degree = V.ufl_element().degree() + residual_degree
    variant = "integral"

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


def _dorfler_mark(indicators, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("marking_fraction must lie in (0, 1]")
    local = indicators.dat.data_ro.copy()
    if not np.isfinite(local).all():
        raise RuntimeError("DWR error indicators contain non-finite values")
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
    """Mark cells using an automatically localized dual-weighted residual."""

    def __init__(self, goal_functional, primal=None, enrichment_degree=None):
        if not isinstance(goal_functional, ufl.BaseForm) or goal_functional.arguments():
            raise ValueError("goal_functional must be a 0-form")
        self.goal_functional = goal_functional
        self._primal = primal
        self._enrichment_degree = enrichment_degree
        self._high_space = None
        if primal is not None:
            V = primal.function_space()
            self._high_space = _reconstruct_degree(
                V, V.ufl_element().degree() + enrichment_degree
            )

    def setup(self, primal, options_prefix):
        options = PETSc.Options(options_prefix)
        self._primal = primal
        self._enrichment_degree = options.getInt("dwr_enrichment_degree", 1)
        V = primal.function_space()
        self._high_space = _reconstruct_degree(
            V, V.ufl_element().degree() + self._enrichment_degree
        )

    def reconstruct(self, coefficient_mapping):
        goal = replace(self.goal_functional, coefficient_mapping)
        primal = coefficient_mapping[self._primal]
        return type(self)(goal, primal, self._enrichment_degree)

    def __call__(self, ctx, current_solution):
        return self._mark(ctx, current_solution)

    def _mark(self, ctx, current_solution):
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
        if not np.isfinite(dual_low.dat.data_ro).all():
            raise RuntimeError("DWR low-order dual solution contains non-finite values")

        primal_high = Function(high_space, name="dwr_primal_high")
        primal_high.interpolate(current_solution)
        test, = problem.F.arguments()
        test_high = test.reconstruct(function_space=high_space)
        F_high = replace(problem.F, {current_solution: primal_high, test: test_high})
        bcs_high = [bc.reconstruct(V=high_space, indices=bc._indices)
                    for bc in problem.bcs]
        high_problem = NonlinearVariationalProblem(F_high, primal_high, bcs_high)
        primal_solver = NonlinearVariationalSolver(
            high_problem,
            options_prefix=prefix + "dwr_primal_",
        )
        primal_solver.solve()
        if not np.isfinite(primal_high.dat.data_ro).all():
            raise RuntimeError("DWR enriched primal solution contains non-finite values")

        goal_high = replace(self.goal_functional, {current_solution: primal_high})
        dual_high = Function(high_space, name="dwr_dual_high")
        direction_high = TrialFunction(high_space)
        dual_test_high = TestFunction(high_space)
        jacobian_high = derivative(F_high, primal_high, direction_high)
        goal_derivative_high = derivative(goal_high, primal_high, dual_test_high)
        dual_problem = LinearVariationalProblem(
            adjoint(jacobian_high), goal_derivative_high, dual_high,
            bcs=_homogeneous_bcs(problem.bcs, high_space),
        )
        dual_solver = LinearVariationalSolver(
            dual_problem,
            options_prefix=prefix + "dwr_dual_",
        )
        dual_solver.solve()
        if not np.isfinite(dual_high.dat.data_ro).all():
            raise RuntimeError("DWR enriched dual solution contains non-finite values")

        dual_error = dual_high - dual_low
        indicators = _residual_indicators(
            problem.F, dual_error, residual_degree, prefix
        )
        return _dorfler_mark(indicators, marking_fraction)


def dwr_marking_callback(goal_functional: ufl.BaseForm):
    """Construct a dual-weighted residual cell-marking callback.

    Parameters
    ----------
    goal_functional
        A scalar UFL 0-form depending on the primal solution.

    Returns
    -------
    DWRMarkingCallback
        A callback suitable for ``solve(..., marking_callback=...)``.

    Notes
    -----
    Options are read from the active solver's PETSc options prefix. The
    supported options are ``dwr_enrichment_degree`` (default 1),
    ``dwr_residual_degree`` (default 1), and ``dwr_marking_fraction``
    (default 0.5). The auxiliary solvers use the ``dwr_primal_``,
    ``dwr_dual_``, ``dwr_cell_``, and ``dwr_facet_`` sub-prefixes.
    """
    return DWRMarkingCallback(goal_functional)
