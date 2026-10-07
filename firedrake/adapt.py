from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import petsctools
import ufl
from finat.ufl import BrokenElement, FiniteElement
from petsctools.options import DefaultOptionSet, get_default_options
from ufl import avg, dS, ds, dx, inner, replace

from firedrake.assemble import assemble
from firedrake.cython import dmcommon
from firedrake.cython import mgimpl as impl
from firedrake.exceptions import ConvergenceError
from firedrake.utils import IntType
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.logging import RED, warning
from firedrake.mesh import Mesh, DISTRIBUTION_PARAMETERS_NOOP
from firedrake.netgen import _snap_to_netgen, _curve_netgen_mesh
from firedrake.petsc import PETSc
from firedrake.solving_utils import _SNESContext
from firedrake.ufl_expr import TestFunction, TrialFunction, derivative
from firedrake.variational_solver import (LinearVariationalProblem,
                                          LinearVariationalSolver,
                                          NonlinearVariationalSolver)


# PETSc's DMAdaptFlag value requesting refinement, for the adapt label.
DM_ADAPT_REFINE = 1

ADAPT_LABEL = "_adaptive_dmplex_adapt"


def _adapt_marked_cells(mesh, cell_marker):
    """Refine the cells of ``mesh`` marked by ``cell_marker`` and return the refined DMPlex."""
    dm = mesh.topology_dm
    ncoarse = mesh.cell_set.size

    # Save the transform, so that the refined DMPlex can tell which of its
    # points came from which point of ``dm``.
    dm.setSaveTransform()

    with PETSc.Log.Event("AdaptiveRefine: mark cells"):
        dm.createLabel(ADAPT_LABEL)
        adapt_label = dm.getLabel(ADAPT_LABEL)
        adapt_indicator = np.zeros(cell_marker.dat.data_ro_with_halos.shape, dtype=IntType)
        adapt_indicator[:ncoarse] = cell_marker.dat.data_ro.real > 0
        dmcommon.mark_points_with_function_array(
            dm, cell_marker.function_space().dm.getLocalSection(), 0,
            adapt_indicator, adapt_label, DM_ADAPT_REFINE,
        )

    # DMPlexTransform reads its type without a prefix. Keep a type that the
    # user has set, because inserted_options deletes what it inserts.
    options = PETSc.Options()
    transform_type = "dm_plex_transform_type"
    parameters = {} if transform_type in options else {transform_type: "refine_sbr"}
    try:
        with petsctools.inserted_options(parameters=parameters, options_prefix=""):
            with PETSc.Log.Event("AdaptiveRefine: adaptLabel"):
                new_dm = dm.adaptLabel(ADAPT_LABEL)
    finally:
        # Ensure the temporary label is removed even if adaptation fails
        dm.removeLabel(ADAPT_LABEL)

    # The transform propagates every label, including the temporary adapt
    # label and the coarse mesh's stale pyop2_core/owned/ghost point
    # classification. Mesh() skips recomputing that classification if it's
    # already present, so it must be dropped here to force a fresh one for
    # the new mesh's own point count and distribution.
    for label in ("pyop2_core", "pyop2_owned", "pyop2_ghost", ADAPT_LABEL):
        if new_dm.hasLabel(label):
            new_dm.removeLabel(label)

    return new_dm


def _copy_adaptive_refinement_metadata(source_mesh, target_mesh):
    """Copy mesh-construction metadata from a mesh onto its adaptively-derived successor."""
    target_mesh._distribution_parameters = dict(source_mesh._distribution_parameters)
    target_mesh._did_reordering = source_mesh._did_reordering
    target_mesh._tolerance = source_mesh.tolerance


def refine_marked_elements(mesh, cell_marker):
    """Adaptively refine a mesh using a DG0 marking function.

    Positive integer marker values request repeated refinement of the
    corresponding cells. The vertices of a Netgen mesh are snapped onto its
    geometry after each round, and the coordinates are curved to their
    original degree at the end.

    Parameters
    ----------
    mesh
        The mesh to refine.
    cell_marker
        A DG0 `~firedrake.function.Function` on ``mesh``: cells with a
        positive value ``n`` are refined ``n`` times.

    Returns
    -------
    MeshGeometry
        The adaptively refined mesh, with ``_adaptive_parent`` set to
        ``mesh`` and ``_adaptive_fine_to_coarse_points`` set to the DMPlex
        point of ``mesh`` that each of its DMPlex points was refined from.

    """
    with cell_marker.dat.vec_ro as v:
        _, num_refinements = v.max()
    # Always run at least one adaptation pass, even when no cell is marked,
    # so that a fresh mesh (with its own cell maps) is produced uniformly.
    num_refinements = max(int(np.rint(num_refinements)), 1)

    current_mesh = mesh
    current_mark = cell_marker
    fine_to_coarse_points = np.arange(*mesh.topology_dm.getChart(), dtype=IntType)
    is_netgen = hasattr(mesh, "netgen_mesh")
    for ref in range(num_refinements):
        new_dm = _adapt_marked_cells(current_mesh, current_mark)
        if is_netgen:
            ngmesh = _snap_to_netgen(new_dm, mesh.netgen_mesh)
        fine_to_coarse_points = impl.compose_points(
            fine_to_coarse_points, impl.transform_source_points(new_dm))
        with PETSc.Log.Event("AdaptiveRefine: Mesh()"):
            current_mesh = Mesh(
                new_dm,
                dim=mesh.geometric_dimension,
                reorder=False,
                distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
                comm=mesh.comm,
                tolerance=mesh.tolerance,
            )
        if is_netgen:
            current_mesh.netgen_mesh = ngmesh
            current_mesh.netgen_flags = mesh.netgen_flags
        if ref < num_refinements - 1:
            with PETSc.Log.Event("AdaptiveRefine: re-mark"):
                # A cell asking for n refinements stays marked until n rounds
                # have happened, so its descendants inherit n minus the number
                # of rounds so far.
                _, fine_to_coarse = impl.coarse_to_fine_cells(mesh, current_mesh, fine_to_coarse_points)
                ancestor = fine_to_coarse[:, 0]
                refined = ancestor >= 0
                current_mark = Function(FunctionSpace(current_mesh, "DG", 0))
                current_mark.dat.data_wo[refined] = \
                    cell_marker.dat.data_ro[ancestor[refined]] - (ref + 1)

    final_mesh = current_mesh
    if is_netgen:
        coordinates = mesh.coordinates.function_space()
        with PETSc.Log.Event("AdaptiveRefine: recurve netgen coords"):
            final_mesh = _curve_netgen_mesh(final_mesh, coordinates.ufl_element().degree(),
                                            cg_field=not coordinates.finat_element.is_dg())

    final_mesh._adaptive_parent = mesh
    final_mesh._adaptive_fine_to_coarse_points = fine_to_coarse_points
    _copy_adaptive_refinement_metadata(mesh, final_mesh)
    return final_mesh


def _replace_arguments(form, *arguments):
    return replace(form, dict(zip(form.arguments(), arguments)))


def _both(expr):
    return expr("+") + expr("-")


def _residual_indicators(F, dual_error, residual_degree, options_prefix):
    """Compute one dual-weighted residual error indicator per cell.

    The residual of the primal solution ``u_h`` is written as a sum of
    cell and facet integrals

    .. math::

        F(u_h; v) = \\sum_K (R_K, v)_K + (R_{\\partial K}, v)_{\\partial K},

    where ``R_K`` is a polynomial on the cell ``K`` and ``R_dK`` is a
    polynomial on each facet of ``K``. This representation is computed
    from the form ``F``, so the strong form of the equation is not needed.
    Two sets of local problems give the residuals. Test functions that are
    a cell bubble times a polynomial vanish on all facets, so testing ``F``
    against them gives ``R_K``. Test functions that are a facet bubble
    times a polynomial then give ``R_dK`` from the remainder
    ``F(u_h; v) - (R_K, v)_K``. The indicator of a cell ``K`` is

    .. math::

        \\eta_K = \\left| (R_K, z - z_h)_K
        + (R_{\\partial K}, z - z_h)_{\\partial K} \\right|,

    where the contribution of an interior facet is the average of the
    contributions from its two sides.

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

    References
    ----------
    Rognes, M. E. and Logg, A., 2013: "Automated goal-oriented error control
    I: Stationary variational problems". https://doi.org/10.1137/10081962X
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
        cell_problem, options_prefix=options_prefix + "goal_cell_",
    )
    cell_solver.solve()

    # Testing the remainder against facet bubbles isolates the facet residual.
    cone_space = FunctionSpace(mesh, "FB", dim, variant=variant)
    cone = Function(cone_space).assign(1)
    cell = mesh.ufl_cell()
    element = BrokenElement(FiniteElement("FB", cell=cell,
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
        facet_problem, options_prefix=options_prefix + "goal_facet_",
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


@dataclass(frozen=True)
class GoalEstimate:
    """The dual-weighted residual error estimate on one mesh.

    Attributes
    ----------
    num_dofs
        The number of degrees of freedom of the primal space.
    goal
        The goal ``J(u_h)``.
    discretisation_error
        The discretisation error ``rho(u_h; z - z_h)``.
    solver_error
        The solver error ``rho(u_h; z_h)``.
    true_error
        The true error ``J(u) - J(u_h)``, or `None` if the exact solution is
        not known.
    """
    num_dofs: int
    goal: float
    discretisation_error: float
    solver_error: float
    true_error: float | None = None

    @property
    def error_estimate(self) -> float:
        """The estimate ``eta`` of ``J(u) - J(u_h)``."""
        return self.discretisation_error + self.solver_error

    @property
    def effectivity_index(self) -> float | None:
        """The ratio of the error estimate to the true error, or `None` if it is undefined."""
        if not self.true_error:
            return None
        return self.error_estimate / self.true_error


class GoalOrientedMarker:
    """A context that marks cells for goal-oriented adaptive refinement.

    This context refines the mesh targeting a localized error estimate
    in a scalar quantity of interest, the goal functional ``J``. It
    estimates the error ``J(u) - J(u_h)`` with the dual-weighted residual
    method, which applies to any variational problem without
    problem-specific derivations.

    Pass an instance as the ``marking_callback`` of a
    `NonlinearVariationalSolver` and set ``snes_adapt_sequence``. The solver
    calls the instance on each mesh in the sequence. The instance moves its
    goal functional to that mesh, records an error estimate in
    ``estimates``, and returns the cells to refine, or `None` to stop the
    adaptation once the estimate meets the tolerances.

    Parameters
    ----------
    goal_functional
        A scalar UFL 0-form that depends on the solution of the problem.
    exact_solution
        An optional UFL expression for the exact solution, which gives the
        true error and the effectivity index of each estimate.

    Attributes
    ----------
    goal_functional
        The goal functional on the most recently marked mesh.
    exact_solution
        The exact solution on the most recently marked mesh, or `None`.
    estimates
        A `GoalEstimate` for each marked mesh, from the coarsest to the finest.

    Notes
    -----
    The marker reads these options from the options prefix of the solver
    that calls it:

    ``goal_atol`` (default 1e-50) and ``goal_rtol`` (default 0)
        Adaptation stops once
        ``|error_estimate| < max(goal_atol, goal_rtol * |J(u_h)|)``. With the
        default tolerances, adaptation runs for the whole
        ``snes_adapt_sequence``.
    ``goal_marking_fraction`` (default 0.5)
        The fraction of the estimated error that the marked cells must
        contain. Larger values refine more cells at each step.
    ``goal_enrichment_degree`` (default 1) and ``goal_residual_degree`` (default 1)
        The increases in polynomial degree that the error estimate uses.
        Larger values give a more accurate estimate at a higher cost.
    ``goal_monitor`` (default off)
        Print each estimate.

    The marker solves the problem again in a space of higher degree,
    with the options of the solver and the ``goal_enriched_`` options, which
    take precedence. The ``goal_cell_`` and ``goal_facet_`` options configure
    the local solves that distribute the estimate over the cells. The error
    estimate requires solves with the transpose of the Jacobian, so the
    preconditioners of the solver and of the ``goal_enriched_`` solver must
    implement ``applyTranspose``.

    Examples
    --------
    Refine at most three times, or until the estimate is below 0.1% of the
    goal::

        marker = GoalOrientedMarker(u * dx)
        solver = NonlinearVariationalSolver(
            problem,
            marking_callback=marker,
            solver_parameters={"snes_adapt_sequence": 3,
                               "adaptor_criterion": "refine",
                               "goal_rtol": 1e-3},
        )
        u_adapted = solver.solve()
        for estimate in marker.estimates:
            print(estimate.num_dofs, estimate.error_estimate)

    References
    ----------
    Rognes, M. E. and Logg, A., 2013: "Automated goal-oriented error control
    I: Stationary variational problems". https://doi.org/10.1137/10081962X
    """

    def __init__(self, goal_functional: ufl.Form,
                 exact_solution: ufl.classes.Expr | None = None):
        if not isinstance(goal_functional, ufl.Form) or goal_functional.arguments():
            raise ValueError("goal_functional must be a 0-form")
        self.goal_functional = goal_functional
        self.exact_solution = exact_solution
        self.estimates: list[GoalEstimate] = []

    def _estimate_error(self, problem, current_solution: Function,
                        dual_low: Function, dual_error: ufl.classes.Expr,
                        options_prefix: str) -> GoalEstimate:
        """Estimate ``J(u) - J(u_h)``, and report on it.

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
        The error estimate on the current mesh.
        """
        goal = assemble(self.goal_functional)
        true_error = None
        if self.exact_solution is not None:
            exact_goal = assemble(replace(self.goal_functional,
                                          {current_solution: self.exact_solution}))
            true_error = exact_goal - goal
        num_dofs = current_solution.function_space().dim()
        discretisation_error = assemble(_replace_arguments(problem.F, -dual_error))
        solver_error = assemble(_replace_arguments(problem.F, -dual_low))
        estimate = GoalEstimate(
            num_dofs=num_dofs,
            goal=goal,
            discretisation_error=discretisation_error,
            solver_error=solver_error,
            true_error=true_error,
        )

        if abs(estimate.solver_error) > abs(estimate.discretisation_error):
            warning(RED % ("DWR: the solver error exceeds the discretisation error. "
                           "Tighten the solver tolerances."))

        if PETSc.Options(options_prefix).getBool("goal_monitor", False):
            report = [("degrees of freedom", estimate.num_dofs),
                      ("goal J(u_h)", estimate.goal),
                      ("discretisation error rho(u_h; z-z_h)", estimate.discretisation_error),
                      ("solver error rho(u_h; z_h)", estimate.solver_error),
                      ("error estimate eta", estimate.error_estimate),
                      ("true error J(u) - J(u_h)", estimate.true_error),
                      ("effectivity index", estimate.effectivity_index)]
            for label, value in report:
                if value is not None:
                    PETSc.Sys.Print(f"    DWR {label:<38s}{value: 15.8e}",
                                    comm=current_solution.comm)
        return estimate

    def __call__(self, ctx: _SNESContext, current_solution: Function) -> Function | None:
        from firedrake.mg.ufl_utils import refine

        problem = ctx._problem
        V = current_solution.function_space()
        if self.goal_functional.ufl_domain() is not V.mesh():
            # The solver has refined the mesh since the previous call.
            mapping = ctx._coefficient_mapping
            self.goal_functional = refine(self.goal_functional, refine, coefficient_mapping=mapping)
            self.exact_solution = refine(self.exact_solution, refine, coefficient_mapping=mapping)
        # Refined contexts have a prefix for their multigrid level, so read the
        # options of the SNES, which all contexts share.
        prefix = ctx.snes.getOptionsPrefix() or ""
        options = PETSc.Options(prefix)
        enrichment_degree = options.getInt("goal_enrichment_degree", 1)
        residual_degree = options.getInt("goal_residual_degree", 1)
        marking_fraction = options.getReal("goal_marking_fraction", 0.5)
        high_degree = V.ufl_element().degree() + enrichment_degree
        high_space = V.reconstruct(degree=high_degree)

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

        # Use options not prefixed by goal_ as defaults for the goal_enriched_ solver
        # This filters out the options given to goal_cell_ and goal_facet_ subsolvers
        parameters = get_default_options(DefaultOptionSet(prefix, ("goal_",)))
        parameters.pop("snes_adapt_sequence", None)

        primal_solver = NonlinearVariationalSolver(
            high_problem,
            options_prefix=prefix + "goal_enriched_",
            solver_parameters=parameters,
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace,
        )
        primal_solver.solve()

        goal_high = replace(self.goal_functional, {current_solution: primal_high})
        dual_high = Function(high_space, name="dwr_dual_high")
        rhs_high = assemble(derivative(goal_high, primal_high), bcs=high_problem.bcs)
        primal_solver._ctx.solve_jacobian(rhs_high, dual_high, transpose=True)

        dual_error = dual_high - dual_low
        estimate = self._estimate_error(
            problem, current_solution, dual_low, dual_error, prefix
        )
        self.estimates.append(estimate)
        atol = options.getReal("goal_atol", 1e-50)
        rtol = options.getReal("goal_rtol", 0.0)
        if abs(estimate.error_estimate) < max(atol, rtol * abs(estimate.goal)):
            # Returning None tells PETSc to stop adapting.
            return None

        indicators = _residual_indicators(
            problem.F, dual_error, residual_degree, prefix
        )
        return _dorfler_mark(indicators, marking_fraction)
