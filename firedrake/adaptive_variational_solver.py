import numbers
from dataclasses import dataclass
from typing import ClassVar

from petsctools import OptionsManager

from firedrake.assemble import assemble
from firedrake.petsc import PETSc
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.ufl_expr import TestFunction, TrialFunction, derivative, action, adjoint
from firedrake.variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                          LinearVariationalProblem, LinearVariationalSolver)
from firedrake.solving import solve
from firedrake.output import VTKFile
from firedrake.logging import RED

from firedrake.preconditioners.pmg import PMGPC
from firedrake.mg.utils import get_level
from firedrake.mg.ufl_utils import refine
from firedrake.mg.adaptive_hierarchy import AdaptiveMeshHierarchy
from firedrake.mg.adaptive_transfer_manager import AdaptiveTransferManager

from finat.ufl import BrokenElement, FiniteElement
from ufl import avg, dx, ds, dS, inner, replace
import ufl


__all__ = ["GoalAdaptiveSolverBase",
           "SteadyGoalAdaptiveSolver",
           "GoalAdaptiveNonlinearVariationalSolver",
           "vtk_output_callback"]


@dataclass(frozen=True)
class GoalAdaptiveOptions:
    """Options for goal-adaptive solvers.

    Parameters
    ----------
    tolerance
        Terminate the adaptive loop when ``|eta_h| < tolerance``.
    max_it
        Maximum number of SOLVE–ESTIMATE–MARK–REFINE cycles.  The loop also
        terminates early if the error estimate falls below the requested tolerance.
        Defaults to ``10``.
    dorfler_alpha
        Threshold parameter for Dörfler (bulk) marking: cells whose local error
        indicator exceeds ``dorfler_alpha * max_indicator`` are marked for
        refinement.  Must lie in ``(0, 1]``.  Larger values mark fewer cells and
        produce more targeted (but potentially slower-converging) refinement.
        Defaults to ``0.5``.
    primal_extra_degree
        Extra polynomial degree used when solving the primal problem in an
        enriched space (only relevant when ``use_adjoint_residual=True``).
        The enriched-space degree is ``degree + primal_extra_degree``.
        Defaults to ``1``.
    dual_extra_degree
        Extra polynomial degree used when solving the dual (adjoint) problem
        in an enriched space.  The enriched-space degree is
        ``degree + dual_extra_degree``.  Defaults to ``1``.
    cell_residual_extra_degree
        Extra polynomial degree for the DG space used to represent cell
        residuals during error indicator computation.  The DG space degree is
        ``degree + cell_residual_extra_degree``.  Defaults to ``1``.
    facet_residual_extra_degree
        Extra polynomial degree for the broken facet-bubble space used to
        represent facet residuals during error indicator computation.
        The space degree is ``degree + facet_residual_extra_degree``.
        Defaults to ``1``.
    use_adjoint_residual
        If ``True``, both the primal residual
        :math:`\\rho(u_h; z - z_h)` and the adjoint residual
        :math:`\\rho^*(z_h; u - u_h)` are used to form the error estimate
        :math:`\\frac{1}{2}(\\rho + \\rho^*)`, which gives a remainder term
        that is cubic in the errors z - z_h and u - u_h.
        This requires four PDE solves per iteration (primal and dual each at two degrees).
        If ``False`` (the default), only the primal residual is used, requiring
        only two PDE solves per iteration (primal at degree :math:`p` and dual
        at degree :math:`p + \\text{dual\\_extra\\_degree}`). This means
        that the remainder term is instead quadratic in the errors
        z - z_h and u - u_h.
        Defaults to ``False``.
    primal_low_method
        How to obtain the low-degree primal solution when
        ``use_adjoint_residual=True``.  Options:

        * ``"interpolate"`` (default) – nodally interpolate the enriched-space
          solution into the base space.
        * ``"project"`` – :math:`L^2`-project the enriched-space solution into
          the base space.
        * ``"solve"`` – solve the primal problem independently in the base
          space; the enriched-space solution is used only for the error estimate.

    dual_low_method
        How to obtain the low-degree dual solution used in the error estimate.
        Options:

        * ``"interpolate"`` (default) – nodally interpolate the enriched-space
          dual solution into the base space.
        * ``"project"`` – :math:`L^2`-project the enriched-space dual solution
          into the base space.
        * ``"solve"`` – solve the dual problem independently in the base space;
          this is the most expensive option but produces the most accurate solver-error
          estimate.

    verbose
        If ``True`` (the default), print progress information at each
        iteration via :func:`PETSc.Sys.Print`.
    """

    tolerance: float = 1e-4
    max_it: int = 10
    dorfler_alpha: float = 0.5
    primal_extra_degree: int = 1
    dual_extra_degree: int = 1
    cell_residual_extra_degree: int = 1
    facet_residual_extra_degree: int = 1
    use_adjoint_residual: bool = False
    primal_low_method: str = "interpolate"
    dual_low_method: str = "interpolate"
    verbose: bool = True

    # Solver parameters for cell/facet bubble projections
    sp_cell: ClassVar[dict] = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "jacobi",
    }
    sp_facet: ClassVar[dict] = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "jacobi",
    }


class GoalAdaptiveSolverBase:
    """Base class for goal-adaptive solvers.

    Owns the SOLVE→ESTIMATE→MARK→REFINE loop and the Dörfler marking
    strategy.  Subclasses must implement :meth:`solve_and_estimate`,
    :meth:`compute_error_indicators`, and :meth:`refine_problem`.

    Parameters
    ----------
    goal_adaptive_options
        A ``GoalAdaptiveOptions`` instance, or a dict of keyword
        arguments to construct one.  Defaults to ``{}``.
    exact_goal
        Exact value of the goal functional (or eigenvalue).  Optional; used
        by subclasses to compute efficiency indices.
    """

    def __init__(self, base_mesh,
                 goal_adaptive_options: "GoalAdaptiveOptions | dict | None" = None,
                 exact_goal=None,
                 post_iteration_callback=None,
                 ):
        if goal_adaptive_options is None:
            goal_adaptive_options = {}
        self.options = self._make_options(goal_adaptive_options)
        self.goal_exact = exact_goal
        self.post_iteration_callback = post_iteration_callback

        self.Ndofs_vec: list[int] = []
        self.eta_vec: list[float] = []
        self.etah_vec: list[float] = []

        # Set up an AdaptiveMeshHierarchy for every mesh of the problem.
        # For a MeshSequenceGeometry, iterating over it yields the component meshes;
        # we also need a separate AMH for the sequence geometry itself.
        meshes = {*base_mesh, base_mesh}  # component meshes + the sequence geometry (same object for a regular mesh)
        for mesh in meshes:
            mh, level = get_level(mesh)
            if mh is None:
                amh = AdaptiveMeshHierarchy(mesh)
            else:
                amh = AdaptiveMeshHierarchy(mh[0])
                for m in mh[1:level+1]:
                    amh.add_mesh(m)

        amh, _ = get_level(base_mesh)
        self.amh = amh
        self.base_levels = len(amh)
        self.atm = AdaptiveTransferManager()

    def _make_options(self, d):
        """Construct a ``GoalAdaptiveOptions`` from a dict or pass through as-is.  Override in subclasses."""
        if isinstance(d, GoalAdaptiveOptions):
            return d
        # FIXME
        return GoalAdaptiveOptions(**d["goal_adaptive"])

    def get_error_estimate(self):
        return self.etah_vec[-1]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def solve(self):
        """Run the adaptive SOLVE→ESTIMATE→MARK→REFINE loop to convergence."""
        for it in range(self.options.max_it):
            try:
                self.step(it=it)
            except StopIteration:
                break

    def step(self, it):
        """Execute one SOLVE→ESTIMATE→MARK→REFINE cycle.

        Parameters
        ----------
        it
            Current iteration index (mesh level).

        Raises
        ------
        StopIteration
            Raised (instead of returning) when the error estimate falls below
            ``tolerance`` or when ``it`` reaches ``max_it - 1``.
            :meth:`solve` catches this automatically; callers driving the loop
            manually with :meth:`step` must handle it themselves.
        """
        self.print(f"---------------------------- [MESH LEVEL {it}] ----------------------------")
        # SOLVE + ESTIMATE
        eta_h, eta = self.solve_and_estimate()
        self.post_iteration(it)
        if abs(eta_h) < self.options.tolerance:
            self.print("Error estimate below tolerance, finished.")
            raise StopIteration
        elif it == self.options.max_it - 1:
            self.print(f"Maximum iteration ({self.options.max_it}) reached. Exiting.")
            raise StopIteration
        # MARK
        self.print("Computing local refinement indicators eta_K ...")
        eta_cell = self.compute_error_indicators()
        self.compute_efficiency_indices(eta_cell, eta_h, eta)
        markers = self.set_adaptive_cell_markers(eta_cell)
        # REFINE
        self.print("Transferring problem to new mesh ...")
        self.refine_problem(markers)

    def post_iteration(self, it: int):
        """Hook called after SOLVE+ESTIMATE, before convergence check.
           Invokes the user-supplied ``post_iteration_callback``, if any."""
        if self.post_iteration_callback is not None:
            self.post_iteration_callback(self, it)

    # ------------------------------------------------------------------
    # Common machinery (mark + refine + efficiency)
    # ------------------------------------------------------------------

    def set_adaptive_cell_markers(self, eta_cell):
        """Mark cells for refinement using Dörfler marking.

        Parameters
        ----------
        eta_cell
            Cell-wise error indicators (DG0 Function).

        Returns
        -------
        Function
            A DG0 Function with value 1 on cells selected for refinement.
        """
        # NOTE this is not quite Dorfler marking
        # For Dorfler marking we need to implement a parallel sort
        markers = Function(eta_cell.function_space())
        for m, e in zip(markers.subfunctions, eta_cell.subfunctions):
            with e.dat.vec_ro as evec:
                _, emax = evec.max()
            threshold = self.options.dorfler_alpha * emax
            m.dat.data_wo[e.dat.data_ro > threshold] = 1
        return markers

    def refine_problem(self, markers, coef_map=None):
        """Refine the mesh and reconstruct the problem on the new mesh.

        Parameters
        ----------
        markers
            DG0 Function with value 1 on cells to refine.
        """
        for marker in markers.subfunctions:
            mesh = marker.function_space().mesh()
            new_mesh = mesh.refine_marked_elements(marker)
            amh, _ = get_level(mesh)
            amh.add_mesh(new_mesh)

        # Reconstruct MeshSequence with the refined meshes
        mesh = self.amh[-1]
        if len(mesh) > 1:
            new_mesh = type(mesh)([get_level(m)[0][-1] for m in mesh])
            self.amh.add_mesh(new_mesh)
        if coef_map is None:
            coef_map = {}
        self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)

    def compute_efficiency_indices(self, eta_cell, eta_h, eta):
        """Hook called after marking.  Default no-op; override in subclasses."""
        pass

    def print(self, *args, **kwargs):
        if self.options.verbose:
            PETSc.Sys.Print(*args, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    def solve_and_estimate(self):
        """Solve the PDE(s) and compute a global error estimate.

        Must store any state needed by :meth:`compute_error_indicators`.

        Returns
        -------
        tuple[float, float | None]
            ``(eta_h, eta)`` where ``eta_h`` is the error estimate and ``eta``
            is the true error if an exact solution/goal was supplied, else
            ``None``.
        """
        raise NotImplementedError

    def compute_error_indicators(self):
        """Compute cell-wise error indicators using stored solver state.

        Returns
        -------
        Function
            A DG0 Function of cell-wise indicators (absolute values).
        """
        raise NotImplementedError


class SteadyGoalAdaptiveSolver(GoalAdaptiveSolverBase):
    """Intermediate base for steady-state goal-adaptive solvers.

    Adds the effectivity-index tracking that is specific to steady problems
    (where the notion of a scalar goal value and a true error make sense).
    The efficiency vectors are populated by :meth:`compute_efficiency_indices`,
    which is called automatically from :meth:`~GoalAdaptiveSolverBase.step`.

    Attributes
    ----------
    eta_cell_sum_vec : list[float]
        Sum of all cell indicators at each refinement level.
    eff1_vec : list[float]
        Effectivity index ``|eta_h / eta|`` at each level (only when
        ``exact_goal`` is provided).
    eff2_vec : list[float]
        Localisation efficiency ``|sum(eta_K) / eta|`` (only when
        ``exact_goal`` is provided).
    eff3_vec : list[float]
        Ratio ``sum(eta_K) / eta_h`` when no exact goal is available.
    """

    def __init__(self, base_mesh, goal_adaptive_options=None, exact_goal=None, post_iteration_callback=None):
        super().__init__(base_mesh, goal_adaptive_options, exact_goal, post_iteration_callback)
        self.eta_cell_sum_vec: list[float] = []
        self.eff1_vec: list[float] = []
        self.eff2_vec: list[float] = []
        self.eff3_vec: list[float] = []

    def compute_efficiency_indices(self, eta_cell, eta_h, eta):
        """Compute and log effectivity and localisation efficiency indices."""
        with eta_cell.dat.vec as evec:
            eta_cell_total = abs(evec.sum())

        self.eta_cell_sum_vec.append(eta_cell_total)
        self.print(f'{"Sum of refinement indicators:":40s}{eta_cell_total: 15.12e}')

        if eta is not None:
            eff1 = abs(eta_h / eta)
            eff2 = abs(eta_cell_total / eta)
            self.eff1_vec.append(eff1)
            self.eff2_vec.append(eff2)
            self.print(f'{"Effectivity index:":40s}{eff1: 15.12f}')
            self.print(f'{"Localisation efficiency:":40s}{eff2: 15.12f}')
        else:
            eff3 = eta_cell_total / abs(eta_h)
            self.eff3_vec.append(eff3)
            self.print(f'{"Localisation efficiency:":40s}{eff3: 15.12f}')


class GoalAdaptiveNonlinearVariationalSolver(SteadyGoalAdaptiveSolver, OptionsManager):
    """Solves a nonlinear variational problem to minimise the error in a
    user-specified goal functional by adaptively refining the mesh using the
    dual-weighted residual (DWR) error estimate.

    All options — both goal-adaptive loop parameters and PETSc solver
    parameters for the inner primal/dual solves — are passed through a
    single ``solver_parameters`` dictionary.  Goal-adaptive parameters are
    distinguished by a ``"goal_adaptive"`` namespace key (which after
    flattening becomes a ``goal_adaptive_`` prefix), e.g.::

        solver_parameters = {
            "goal_adaptive": {
                "tolerance": 1e-4,
                "max_it": 8,
                "dual_low_method": "interpolate",
                "verbose": False,
            },
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

    Parameters
    ----------
    problem
        The variational problem defined on the initial (coarse) mesh.
    goal_functional
        The goal functional — a zero-form in terms of the primal solution.
    solver_parameters
        Unified parameter dictionary.  Keys prefixed by ``goal_adaptive_``
        (or nested under a ``"goal_adaptive"`` sub-dict) configure the
        adaptive loop (see ``GoalAdaptiveOptions``); all other keys are
        passed to the inner :class:`~.NonlinearVariationalSolver` /
        :class:`~.LinearVariationalSolver`.
    options_prefix
        PETSc options prefix, forwarded to ``petsctools.OptionsManager``.
        Allows command-line overrides, e.g.
        ``-mysolve_snes_type ksponly``.
    primal_solver_kwargs
        Extra keyword arguments for the primal :class:`~.NonlinearVariationalSolver`.
    dual_solver_kwargs
        Extra keyword arguments for the dual :class:`~.LinearVariationalSolver`.
    exact_solution
        Exact primal solution (UFL expression or list/tuple for mixed spaces).
        Used to compute the true error for efficiency indices.
    exact_goal
        Exact scalar value of the goal functional.  Used to compute the true
        error when an analytic formula is available.
    post_iteration_callback
        Optional callable ``callback(solver, it)`` invoked after each
        SOLVE+ESTIMATE step (before convergence check and refinement).
        Use this for visualisation or post-processing at each mesh level.
        See :func:`vtk_output_callback` for a ready-made VTK writer.
    """

    _GOAL_PREFIX = "goal_adaptive_"

    def __init__(self,
                 problem: NonlinearVariationalProblem,
                 goal_functional: ufl.BaseForm,
                 *,
                 solver_parameters: dict | None = None,
                 options_prefix: str | None = None,
                 primal_solver_kwargs: dict | None = None,
                 dual_solver_kwargs: dict | None = None,
                 exact_solution: ufl.classes.Expr | None = None,
                 exact_goal: ufl.classes.Expr | None = None,
                 post_iteration_callback=None,
                 ):
        if not (isinstance(goal_functional, ufl.BaseForm) and len(goal_functional.arguments()) == 0):
            raise ValueError("goal_functional must be a 0-form")

        if options_prefix is None:
            options_prefix = ""
        base_mesh = problem.u.function_space().mesh()
        SteadyGoalAdaptiveSolver.__init__(self, base_mesh, solver_parameters,
                                          exact_goal=exact_goal,
                                          post_iteration_callback=post_iteration_callback)
        OptionsManager.__init__(self, solver_parameters, options_prefix)

        self.problem = problem
        self.goal_functional = goal_functional
        self.primal_solver_kwargs = primal_solver_kwargs or {}
        self.dual_solver_kwargs = dual_solver_kwargs or {}
        self.u_exact = exact_solution

        self.u_high = None
        # Internal state set by solve_and_estimate, used by compute_error_indicators
        self._u_err = None
        self._z_lo = None
        self._z_err = None

    def _current_solution(self):
        """Return the solution function on the current (finest) mesh."""
        return self.u_high if self.u_high is not None else self.problem.u

    def solve(self):
        """Run the adaptive loop and return the final solution and error estimate.

        Each call continues from the current mesh and solution; history vectors
        (``Ndofs_vec``, ``etah_vec``, etc.) are appended rather than reset.

        Returns
        -------
        tuple[Function, float]
            ``(u_out, error_estimate)`` where ``u_out`` is the solution on the
            finest mesh reached and ``error_estimate`` is the final ``|eta_h|``.
        """
        super().solve()
        return self._current_solution()

    def step(self, it=None):
        """Compute one SOLVE→ESTIMATE→MARK→REFINE step and return the current solution.

        Useful for users who want to inspect or post-process the solution at
        each mesh level without running the full :meth:`solve` loop.

        Parameters
        ----------
        it
            Mesh level index.  If ``None``, inferred from the current mesh
            hierarchy level.

        Returns
        -------
        tuple[Function, float]
            ``(u_out, eta_h)`` — the solution and error estimate on the current
            mesh.  Only returned when refinement was performed.

        Raises
        ------
        StopIteration
            When the error estimate is below ``tolerance`` or the maximum
            iteration count is reached.  Callers must catch this::

                for it in range(solver.options.max_it):
                    try:
                        u, eta = solver.step(it)
                    except StopIteration:
                        break
        """
        if it is None:
            V = self.problem.u.function_space()
            _, it = get_level(V.mesh())
        super().step(it)
        return self._current_solution()

    def solve_and_estimate(self):
        """Solve primal and dual, compute global error estimate."""
        u_err = self.solve_primal()
        z_lo, z_err = self.solve_dual()
        self._u_err = u_err
        self._z_lo = z_lo
        self._z_err = z_err
        eta_h, eta = self.estimate_error(u_err, z_lo, z_err)
        return eta_h, eta

    def solve_primal(self):
        """Solve the primal problem and return the primal error representative.

        When ``use_adjoint_residual=False`` (the default), the primal problem is
        solved once in the base space ``V``.

        When ``use_adjoint_residual=True``, the problem is also solved in an
        enriched space of degree ``degree + primal_extra_degree``.  The
        low-degree approximation is obtained via
        ``primal_low_method`` (interpolate / project / solve), and the
        difference ``u_high - u`` is returned as the primal error representative.

        Returns
        -------
        Function or None
            The primal error representative ``u_high - u_h``, or ``None`` when
            ``use_adjoint_residual=False``.
        """
        F = self.problem.F
        u = self.problem.u
        bcs = self.problem.bcs
        V = self.problem.u.function_space()
        self.Ndofs_vec.append(V.dim())
        primal_options_prefix = self.options_prefix + "primal_"

        def solve_uh():
            self.print(f'Solving primal (degree: {V.ufl_element().degree()}, dofs: {V.dim()}) ...')
            solver = NonlinearVariationalSolver(self.problem, solver_parameters=self.parameters,
                                                options_prefix=primal_options_prefix,
                                                **self.primal_solver_kwargs)
            solver.set_transfer_manager(self.atm)
            solver.solve()
            self.primal_solver = solver

        if self.options.use_adjoint_residual:
            if self.options.primal_low_method == "solve":
                solve_uh()

            # Now solve in higher-order space
            high_degree = V.ufl_element().degree() + self.options.primal_extra_degree
            V_high = reconstruct_degree(V, high_degree)
            u_high = Function(V_high, name="high_order_solution")
            u_high.interpolate(u)

            v_old, = F.arguments()
            v_high = v_old.reconstruct(function_space=V_high)
            F_high = replace(F, {v_old: v_high, u: u_high})
            bcs_high = [bc.reconstruct(V=V_high, indices=bc._indices) for bc in bcs]
            problem_high = NonlinearVariationalProblem(F_high, u_high, bcs_high)

            self.print(f"Solving primal with higher order for error estimate (degree: {high_degree}, dofs: {V_high.dim()}) ...")
            solver = NonlinearVariationalSolver(problem_high, solver_parameters=self.parameters,
                                                options_prefix=primal_options_prefix,
                                                **self.primal_solver_kwargs)
            solver.set_transfer_manager(self.atm)
            solver.solve()
            self.primal_solver = solver

            self.u_high = u_high

            if self.options.primal_low_method == "solve":
                pass
            elif self.options.primal_low_method == "project":
                u.project(u_high)
            elif self.options.primal_low_method == "interpolate":
                u.interpolate(u_high)
            else:
                raise ValueError(f"Unrecognised primal_low_method {self.options.primal_low_method}")
            u_err = u_high - u
        else:
            solve_uh()
            u_err = None
        return u_err

    def solve_dual(self):
        """Solve the dual (adjoint) problem and return the dual solutions.

        The dual problem is always solved in an enriched space of degree
        ``degree + dual_extra_degree``.  A low-degree approximation ``z_lo`` is
        obtained via ``dual_low_method`` (interpolate / project / solve).

        Returns
        -------
        tuple[Function, Function]
            ``(z_lo, z_err)`` where ``z_lo`` is the low-degree dual solution
            (in the same space as the primal ``u``) and ``z_err = z - z_lo``
            is the dual error representative used to weight the residuals.
        """

        def solve_zh(z):
            bcs = self.problem.bcs
            J = self.goal_functional
            F = self.problem.F
            u = self.problem.u
            dual_options_prefix = self.options_prefix + "dual_"

            Z = z.function_space()
            self.print(f"Solving dual (degree: {Z.ufl_element().degree()}, dofs: {Z.dim()}) ...")

            Fz = residual(F, TestFunction(Z))
            dF = derivative(Fz, u, TrialFunction(Z))
            dJ = derivative(J, u, TestFunction(Z))
            a = adjoint(dF)

            if self.u_high is not None:
                a = replace(a, {u: self.u_high})
                dJ = replace(dJ, {u: self.u_high})

            bcs_dual = [bc.reconstruct(V=Z, indices=bc._indices, g=0) for bc in bcs]
            problem = LinearVariationalProblem(a, dJ, z, bcs_dual)
            solver = LinearVariationalSolver(problem, solver_parameters=self.parameters,
                                             options_prefix=dual_options_prefix,
                                             **self.dual_solver_kwargs)
            solver.set_transfer_manager(self.atm)
            solver.solve()

        # Higher-order dual solution
        V = self.problem.u.function_space()
        dual_degree = V.ufl_element().degree() + self.options.dual_extra_degree
        V_dual = reconstruct_degree(V, dual_degree)
        z = Function(V_dual, name="dual_high_order_solution")
        solve_zh(z)

        # Lower-order dual solution
        z_lo = Function(V, name="dual_low_order_solution")
        if self.options.dual_low_method == "solve":
            z_lo.interpolate(z)
            solve_zh(z_lo)
        elif self.options.dual_low_method == "project":
            z_lo.project(z)
        elif self.options.dual_low_method == "interpolate":
            z_lo.interpolate(z)
        else:
            raise ValueError(f"Unrecognised dual_low_method {self.options.dual_low_method}")
        z_err = z - z_lo
        self.z = z
        return z_lo, z_err

    def estimate_error(self, u_err, z_lo, z_err):
        """Compute the global DWR error estimate for the goal functional.

        Computes the primal residual :math:`\\rho(u_h; z - z_h)` and, when
        ``use_adjoint_residual=True``, also the adjoint residual
        :math:`\\rho^*(z_h; u - u_h)`, combining them as
        :math:`\\frac{1}{2}(\\rho + \\rho^*)`.  Also estimates the solver error
        :math:`\\rho(u_h; z_h)`.

        Parameters
        ----------
        u_err
            Primal error representative ``u_high - u_h`` (or ``None`` when
            ``use_adjoint_residual=False``).
        z_lo
            Low-degree dual solution in the base space.
        z_err
            Dual error representative ``z - z_lo``.

        Returns
        -------
        tuple[float, float | None]
            ``(eta_h, eta)`` — the error estimate and the true error
            ``J(u) - J(u_h)`` (or ``None`` if no exact value was supplied).
        """
        J = self.goal_functional
        F = self.problem.F
        u = self.problem.u

        # Primal contribution to error estimator
        primal_err = assemble(residual(F, -z_err))

        # Dual contribution to error estimator
        if self.options.use_adjoint_residual:
            Z = z_lo.function_space()
            dF = derivative(F, u, TrialFunction(Z))
            dJ = derivative(J, u)
            G = action(adjoint(dF), z_lo) - dJ

            dual_err = assemble(residual(G, -u_err))
            discretisation_error = 0.5 * (primal_err + dual_err)
        else:
            discretisation_error = primal_err

        # Estimate of solver error
        solver_error = assemble(residual(F, -z_lo))

        if abs(solver_error) > abs(discretisation_error):
            self.print(RED % 'Warning: solver error estimate greater than discretisation error estimate, refine solver tolerances')

        # Final error estimate
        eta_h = discretisation_error + solver_error
        self.etah_vec.append(eta_h)

        Juh = assemble(J)
        self.print(f'{"Computed goal J(uh):":40s}{Juh:15.12f}')
        self.Juh = Juh
        if self.goal_exact is not None:
            if isinstance(self.goal_exact, numbers.Real):
                Ju = self.goal_exact
            else:
                Ju = assemble(self.goal_exact)
        elif self.u_exact is not None:
            Ju = assemble(replace(J, {u: self.u_exact}))
        else:
            Ju = None

        if Ju is not None:
            eta = Ju - Juh
            self.eta_vec.append(eta)
            self.print(f'{"Exact goal J(u):":40s}{Ju: 15.12f}')
            self.print(f'{"True error, J(u) - J(u_h):":40s}{eta: 15.12e}')
        else:
            eta = None

        if self.options.use_adjoint_residual:
            self.print(f'{"Primal error, rho(u_h; z-z_h):":40s}{primal_err: 15.12e}')
            self.print(f'{"Dual error,  rho*(z_h; u-u_h):":40s}{dual_err: 15.12e}')
            self.print(f'{"Difference":40s}{abs(primal_err-dual_err):19.12e}')
            self.print(f'{"Discretisation error, 0.5(rho + rho*)":40s}{discretisation_error: 15.12e}')
        else:
            self.print(f'{"Discretisation error, rho(u_h; z-z_h)":40s}{discretisation_error: 15.12e}')
        self.print(f'{"Solver error, rho(u_h; z_h):":40s}{solver_error: 15.12e}')
        self.print(f'{"Final error estimate:":40s}{eta_h: 15.12e}')
        return eta_h, eta

    def compute_error_indicators(self):
        """Compute cell-wise DWR error indicators via bubble/cone projections.

        Projects the primal residual :math:`F(u_h; \\cdot)` onto cell-bubble
        and facet-bubble spaces, then weights by the dual error ``z_err``.
        When ``use_adjoint_residual=True``, the adjoint residual is also
        projected and weighted by ``u_err``, and the two contributions are
        averaged.

        Returns
        -------
        Function
            DG0 Function of absolute-value cell-wise indicators :math:`\\eta_K`.
        """
        J = self.goal_functional
        F = self.problem.F
        u = self.problem.u
        u_err = self._u_err
        z_lo = self._z_lo
        z_err = self._z_err
        V = u.function_space()

        mesh = V.mesh().unique()
        dim = mesh.topological_dimension
        cell = mesh.ufl_cell()
        variant = "integral"
        degree = V.ufl_element().degree()
        cell_residual_degree = degree + self.options.cell_residual_extra_degree
        facet_residual_degree = degree + self.options.facet_residual_extra_degree
        # ------------------------------- Primal residual -------------------------------
        # Cell bubbles
        B = FunctionSpace(mesh, "B", dim+1, variant=variant)
        bubbles = Function(B).assign(1)
        # DG space on cell interiors
        if V.value_shape == ():
            DG = FunctionSpace(mesh, "DG", cell_residual_degree, variant=variant)
        else:
            DG = TensorFunctionSpace(mesh, "DG", cell_residual_degree, variant=variant, shape=V.value_shape)
        uc = TrialFunction(DG)
        vc = TestFunction(DG)
        ac = inner(uc, bubbles*vc)*dx
        Lc = residual(F, bubbles*vc)
        # solve for Rcell
        Rcell = Function(DG)
        solve(ac == Lc, Rcell, solver_parameters=self.options.sp_cell)

        # Facet bubbles
        FB = FunctionSpace(mesh, "FB", dim, variant=variant)
        cones = Function(FB).assign(1)
        # Broken facet bubble space
        el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residual_degree+dim, variant=variant))
        if V.value_shape == ():
            Q = FunctionSpace(mesh, el)
        else:
            Q = TensorFunctionSpace(mesh, el, shape=V.value_shape)
        Qtest = TestFunction(Q)
        Qtrial = TrialFunction(Q)
        Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
        af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds
        # solve for Rfacet
        Rhat = Function(Q)
        solve(af == Lf, Rhat, solver_parameters=self.options.sp_facet)
        Rfacet = Rhat/cones

        # Primal error indicators
        DG0 = FunctionSpace(mesh, "DG", degree=0)
        test = TestFunction(DG0)

        eta_primal = assemble(
            inner(inner(Rcell, z_err), test)*dx +
            + inner(avg(inner(Rfacet, z_err)), both(test))*dS +
            + inner(inner(Rfacet, z_err), test)*ds
        )
        with eta_primal.dat.vec as evec:
            evec.abs()

        # ------------------------------- Adjoint residual -------------------------------
        if self.options.use_adjoint_residual:
            # r*(v) = J'(u)[v] - A'_u(u)[v, z] since F = A(u;v) - L(v)
            dF = derivative(F, u, TrialFunction(V))
            dJ = derivative(J, u, TestFunction(V))
            rstar = action(adjoint(dF), z_lo) - dJ

            # dual: project r* -> Rcell*, Rfacet*
            Lc_star = residual(rstar, bubbles*vc)
            Rcell_star = Function(DG)
            solve(ac == Lc_star, Rcell_star, solver_parameters=self.options.sp_cell)

            Lf_star = residual(rstar, Qtest) - inner(Rcell_star, Qtest)*dx
            Rhat_star = Function(Q)
            solve(af == Lf_star, Rhat_star, solver_parameters=self.options.sp_facet)
            Rfacet_star = Rhat_star/cones

            eta_dual = assemble(
                inner(inner(Rcell_star, u_err), test)*dx
                + inner(avg(inner(Rfacet_star, u_err)), both(test))*dS
                + inner(inner(Rfacet_star, u_err), test)*ds
            )
            with eta_dual.dat.vec as evec:
                evec.abs()
            eta_cell = assemble(0.5*(eta_primal + eta_dual))

            with eta_primal.dat.vec as evec:
                self.eta_primal_total = abs(evec.sum())
            with eta_dual.dat.vec as evec:
                self.eta_dual_total = abs(evec.sum())
            self.print(f'{"Sum of primal refinement indicators:":40s}{self.eta_primal_total: 15.12e}')
            self.print(f'{"Sum of dual refinement indicators:":40s}{self.eta_dual_total: 15.12e}')
        else:
            eta_cell = eta_primal

        return eta_cell

    def refine_problem(self, markers):
        """Adaptively refine the mesh and rediscretise the problem on the refined mesh"""
        coef_map = {}
        super().refine_problem(markers, coef_map=coef_map)

        self.goal_functional = refine(self.goal_functional, refine, coefficient_mapping=coef_map)
        if self.u_exact is not None:
            self.u_exact = refine(self.u_exact, refine, coefficient_mapping=coef_map)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def vtk_output_callback(output_dir="./output", run_name="default"):
    """Return a ``post_iteration_callback`` that writes primal and dual solutions to VTK.

    Usage::

        from firedrake import *
        solver = GoalAdaptiveNonlinearVariationalSolver(
            problem, goal_functional,
            solver_parameters={...},
            post_iteration_callback=vtk_output_callback(
                output_dir="./output", run_name="myproblem"
            ),
        )

    Parameters
    ----------
    output_dir
        Directory in which to write the VTK files.
    run_name
        Label prepended to filenames:
        ``<output_dir>/<run_name>/<run_name>_solution_<it>.pvd`` and
        ``<output_dir>/<run_name>/<run_name>_dual_solution_<it>.pvd``.

    Returns
    -------
    callable
        A function ``callback(solver, it)`` suitable for passing to
        ``post_iteration_callback``.
    """
    def _callback(solver, it):
        prefix = f"{output_dir}/{run_name}/{run_name}"
        comm = solver.problem.u.function_space().mesh().comm
        solver.print("Writing (primal) solution ...")
        VTKFile(f"{prefix}_solution_{it}.pvd", comm=comm).write(*solver.problem.u.subfunctions)
        solver.print("Writing (dual) solution ...")
        VTKFile(f"{prefix}_dual_solution_{it}.pvd", comm=comm).write(*solver.z.subfunctions)
    return _callback


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def residual(F, *args):
    """Replace the test function argument of a Form."""
    return replace(F, dict(zip(F.arguments(), args)))


def both(u):
    """Add u on both sides of a facet."""
    return u("+") + u("-")


def reconstruct_degree(V, degree):
    """Reconstruct a FunctionSpace with a different polynomial degree."""
    return V.reconstruct(element=PMGPC.reconstruct_degree(V.ufl_element(), degree))


def reconstruct_bcs(bcs, V):
    """Reconstruct a list of BCs on a different FunctionSpace."""
    return [bc.reconstruct(V=V, indices=bc._indices) for bc in bcs]


def _compute_residual_indicators(F, z_err, options):
    """Compute DG0 cell-wise error indicators via bubble/cone projections.

    This implements the primal-residual indicator

    .. math::

        \\eta_K = \\int_K R_{\\text{cell}} \\cdot z_{\\text{err}}\\,\\mathrm{d}x
                  + \\int_{\\partial K} R_{\\text{facet}} \\cdot z_{\\text{err}}\\,\\mathrm{d}s

    for the residual form ``F`` against the dual error ``z_err``.

    Parameters
    ----------
    F
        Residual as a linear form (test function as its sole argument).
    z_err
        Dual error representative ``z_p - z_h`` (UFL expression or Function).
    options
        ``GoalAdaptiveOptions`` (or subclass) instance for degree
        parameters and solver parameters.

    Returns
    -------
    Function
        DG0 Function of absolute-value cell indicators.
    """
    v, = F.arguments()
    V = v.function_space()
    mesh = V.mesh().unique()
    dim = mesh.topological_dimension
    cell = mesh.ufl_cell()
    variant = "integral"
    degree = V.ufl_element().degree()
    cell_residual_degree = degree + options.cell_residual_extra_degree
    facet_residual_degree = degree + options.facet_residual_extra_degree

    B = FunctionSpace(mesh, "B", dim+1, variant=variant)
    bubbles = Function(B).assign(1)

    if V.value_shape == ():
        DG = FunctionSpace(mesh, "DG", cell_residual_degree, variant=variant)
    else:
        DG = TensorFunctionSpace(mesh, "DG", cell_residual_degree, variant=variant, shape=V.value_shape)
    uc = TrialFunction(DG)
    vc = TestFunction(DG)
    ac = inner(uc, bubbles*vc)*dx
    Lc = residual(F, bubbles*vc)
    Rcell = Function(DG)
    solve(ac == Lc, Rcell, solver_parameters=options.sp_cell)

    FB = FunctionSpace(mesh, "FB", dim, variant=variant)
    cones = Function(FB).assign(1)
    el = BrokenElement(FiniteElement("FB", cell=cell, degree=facet_residual_degree+dim, variant=variant))
    if V.value_shape == ():
        Q = FunctionSpace(mesh, el)
    else:
        Q = TensorFunctionSpace(mesh, el, shape=V.value_shape)
    Qtest = TestFunction(Q)
    Qtrial = TrialFunction(Q)
    Lf = residual(F, Qtest) - inner(Rcell, Qtest)*dx
    af = both(inner(Qtrial/cones, Qtest))*dS + inner(Qtrial/cones, Qtest)*ds
    Rhat = Function(Q)
    solve(af == Lf, Rhat, solver_parameters=options.sp_facet)
    Rfacet = Rhat/cones

    DG0 = FunctionSpace(mesh, "DG", degree=0)
    test = TestFunction(DG0)
    eta_cell = assemble(
        inner(inner(Rcell, z_err), test)*dx
        + inner(avg(inner(Rfacet, z_err)), both(test))*dS
        + inner(inner(Rfacet, z_err), test)*ds
    )
    with eta_cell.dat.vec as evec:
        evec.abs()
    return eta_cell
