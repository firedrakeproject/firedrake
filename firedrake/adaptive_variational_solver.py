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
from ufl import avg, dx, ds, dS, inner, as_vector, replace
import ufl
import numpy as np
from numbers import Integral


__all__ = ["GoalAdaptiveSolverBase",
           "GoalAdaptiveNonlinearVariationalSolver",
           "GoalAdaptiveEigensolver"]


class GoalAdaptiveOptions:
    """Options for goal-adaptive solvers.

    Parameters
    ----------
    dorfler_alpha
        Threshold parameter for Dörfler (bulk) marking: cells whose local error
        indicator exceeds ``dorfler_alpha * max_indicator`` are marked for
        refinement.  Must lie in ``(0, 1]``.  Larger values mark fewer cells and
        produce more targeted (but potentially slower-converging) refinement.
        Defaults to ``0.5``.
    max_iterations
        Maximum number of SOLVE–ESTIMATE–MARK–REFINE cycles.  The loop also
        terminates early if the error estimate falls below the requested tolerance.
        Defaults to ``10``.
    output_dir
        Directory into which VTK output files are written.
        Defaults to ``"./output"``.
    run_name
        Label prepended to output filenames, e.g.
        ``<output_dir>/<run_name>/<run_name>_solution_<it>.pvd``.
        Defaults to ``"default"``.
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
        Not used by :class:`GoalAdaptiveEigensolver`.
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

        Not used by :class:`GoalAdaptiveEigensolver`.
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

        Not used by :class:`GoalAdaptiveEigensolver`.
    write_solution
        Controls when primal and dual solutions are written to VTK files.

        * ``False`` (default) – never write.
        * ``True`` – write at every iteration.
        * A positive integer ``k`` – write every ``k`` iterations.
    verbose
        If ``True`` (the default), print progress information at each
        iteration via :func:`PETSc.Sys.Print`.
    """

    def __init__(self,
                 dorfler_alpha: float = 0.5,
                 max_iterations: int = 10,
                 output_dir: str = "./output",
                 run_name: str = "default",
                 primal_extra_degree: int = 1,
                 dual_extra_degree: int = 1,
                 cell_residual_extra_degree: int = 1,
                 facet_residual_extra_degree: int = 1,
                 use_adjoint_residual: bool = False,
                 primal_low_method: str = "interpolate",
                 dual_low_method: str = "interpolate",
                 write_solution: bool | int = False,
                 verbose: bool = True,
                 ):
        self.dorfler_alpha = dorfler_alpha
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.run_name = run_name
        self.primal_extra_degree = primal_extra_degree
        self.dual_extra_degree = dual_extra_degree
        self.cell_residual_extra_degree = cell_residual_extra_degree
        self.facet_residual_extra_degree = facet_residual_extra_degree
        self.use_adjoint_residual = use_adjoint_residual
        self.primal_low_method = primal_low_method
        self.dual_low_method = dual_low_method
        self.write_solution = write_solution
        self.verbose = verbose

    # Solver parameters
    sp_cell = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "jacobi",
    }
    sp_facet = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "jacobi",
    }


class GoalAdaptiveEigenOptions(GoalAdaptiveOptions):
    """Options for :class:`GoalAdaptiveEigensolver`.

    Extends :class:`GoalAdaptiveOptions` with eigenproblem-specific parameters.

    Parameters
    ----------
    self_adjoint
        If ``True``, the eigenproblem is self-adjoint so the dual eigenproblem
        equals the primal; only one set of eigensolves is performed per
        iteration.  Defaults to ``False``.
    nev
        Number of eigenvalue/eigenvector pairs to compute at each solve.
        The solver picks the one best correlated with the current eigenfunction
        estimate via :func:`match_best`.  Defaults to ``5``.

    Notes
    -----
    The options ``use_adjoint_residual``, ``primal_low_method``, and
    ``dual_low_method`` inherited from :class:`GoalAdaptiveOptions` are not
    used by :class:`GoalAdaptiveEigensolver`.
    """

    def __init__(self, *args, self_adjoint: bool = False, nev: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_adjoint = self_adjoint
        self.nev = nev


class GoalAdaptiveSolverBase:
    """Base class for goal-adaptive solvers.

    Owns the SOLVE→ESTIMATE→MARK→REFINE loop and the Dörfler marking
    strategy.  Subclasses must implement :meth:`solve_and_estimate`,
    :meth:`compute_error_indicators`, and :meth:`refine_problem`.

    Parameters
    ----------
    tolerance
        Terminate the adaptive loop when ``|eta_h| < tolerance``.
    goal_adaptive_options
        Dictionary passed to :meth:`_make_options` to construct an options
        object.  Defaults to ``{}``.
    exact_goal
        Exact value of the goal functional (or eigenvalue) for computing
        efficiency indices.  Optional.
    """

    def __init__(self,
                 tolerance: float,
                 goal_adaptive_options: dict | None = None,
                 exact_goal=None,
                 ):
        if goal_adaptive_options is None:
            goal_adaptive_options = {}
        self.tolerance = tolerance
        self.options = self._make_options(goal_adaptive_options)
        self.goal_exact = exact_goal

        self.Ndofs_vec = []
        self.eta_vec = []
        self.etah_vec = []
        self.eta_cell_sum_vec = []
        self.eff1_vec = []
        self.eff2_vec = []
        self.eff3_vec = []

    def _make_options(self, d):
        """Construct an options object from a dictionary.  Override in subclasses."""
        return GoalAdaptiveOptions(**d)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def solve(self):
        """Run the adaptive refinement loop."""
        for it in range(self.options.max_iterations):
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
        """
        self.print(f"---------------------------- [MESH LEVEL {it}] ----------------------------")
        # SOLVE + ESTIMATE
        eta_h, eta = self.solve_and_estimate()
        self.write_solution(it)
        if abs(eta_h) < self.tolerance:
            self.print("Error estimate below tolerance, finished.")
            raise StopIteration
        elif it == self.options.max_iterations - 1:
            self.print(f"Maximum iteration ({self.options.max_iterations}) reached. Exiting.")
            raise StopIteration
        # MARK
        self.print("Computing local refinement indicators eta_K ...")
        eta_cell = self.compute_error_indicators()
        self.compute_efficiency_indices(eta_cell, eta_h, eta)
        markers = self.set_adaptive_cell_markers(eta_cell)
        # REFINE
        self.print("Transferring problem to new mesh ...")
        self.refine_problem(markers)

    # ------------------------------------------------------------------
    # Common machinery (mark + efficiency)
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

    def compute_efficiency_indices(self, eta_cell, eta_h, eta):
        """Compute and log efficiency indices."""
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
            eff3 = eta_cell_total / eta_h
            self.eff3_vec.append(eff3)
            self.print(f'{"Localisation efficiency:":40s}{eff3: 15.12f}')

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

    def refine_problem(self, markers):
        """Refine the mesh and reconstruct the problem on the new mesh.

        Parameters
        ----------
        markers
            DG0 Function with value 1 on cells to refine.
        """
        raise NotImplementedError

    def write_solution(self, it):
        """Write solution output (VTK etc.).  Default no-op."""
        pass


class GoalAdaptiveNonlinearVariationalSolver(GoalAdaptiveSolverBase):
    """Solves a nonlinear variational problem to minimise the error in a
    user-specified goal functional.  We do this by adaptively refining the mesh
    based on the solution to a dual problem - which links the goal functional
    to the PDE.

    Parameters
    ----------
    problem
        The variational formulation of the PDE defined on the coarse mesh
    goal_functional
        The goal functional defined in terms of the solution to the PDE
    goal_adaptive_options
        An options dictionary to construct a :class:`GoalAdaptiveOptions`
    primal_solver_parameters
        A dictionary of solver parameters for the primal problem
    dual_solver_parameters
        A dictionary of solver parameters for the dual problem.
        Defaults to `primal_solver_parameters`.
    primal_solver_kwargs
        Keyword arguments for the primal :class:`~.NonlinearVariationalSolver`
    dual_solver_kwargs
        Keyword arguments for the dual :class:`~.LinearVariationalSolver`
    exact_solution
        The exact solution to the problem (optional).
        If supplied, it is used to calculate the efficiency of the error estimate
    exact_goal
        The exact value for the goal functional (optional).
        If supplied, it is used to calculate the efficiency of the error estimate.
    """
    def __init__(self,
                 problem: NonlinearVariationalProblem,
                 goal_functional: ufl.BaseForm,
                 tolerance: float,
                 goal_adaptive_options: dict | None = None,
                 primal_solver_parameters: dict | None = None,
                 dual_solver_parameters: dict | None = None,
                 primal_solver_kwargs: dict | None = None,
                 dual_solver_kwargs: dict | None = None,
                 exact_solution: ufl.classes.Expr | None = None,
                 exact_goal: ufl.classes.Expr | None = None,
                 ):
        if not (isinstance(goal_functional, ufl.BaseForm) and len(goal_functional.arguments()) == 0):
            raise ValueError("goal_functional must be a 0-form")

        super().__init__(tolerance, goal_adaptive_options, exact_goal=exact_goal)

        self.problem = problem
        self.goal_functional = goal_functional
        self.sp_primal = primal_solver_parameters
        self.sp_dual = dual_solver_parameters if dual_solver_parameters is not None else primal_solver_parameters
        self.primal_solver_kwargs = primal_solver_kwargs if primal_solver_kwargs is not None else {}
        self.dual_solver_kwargs = dual_solver_kwargs if dual_solver_kwargs is not None else {}
        self.u_exact = as_mixed(exact_solution) if isinstance(exact_solution, (tuple, list)) else exact_solution

        # Set up an AdaptiveMeshHierarchy for every mesh of the problem
        V = problem.u.function_space()
        meshes = set(V.mesh())
        meshes.add(V.mesh())
        for mesh in meshes:
            mh, level = get_level(mesh)
            if mh is None:
                amh = AdaptiveMeshHierarchy(mesh)
            else:
                amh = AdaptiveMeshHierarchy(mh[0])
                for m in mh[1:level+1]:
                    amh.add_mesh(m)

        self.atm = AdaptiveTransferManager()
        self.base_levels = len(amh)

        self.u_high = None
        # Internal state set by solve_and_estimate, used by compute_error_indicators
        self._u_err = None
        self._z_lo = None
        self._z_err = None

    def solve(self):
        """Run the adaptive loop and return the solution and error estimate.

        Returns
        -------
        tuple[Function, float]
            ``(u_out, error_estimate)`` where ``u_out`` is the solution on the
            finest mesh and ``error_estimate`` is the final ``|eta_h|``.
        """
        super().solve()
        u_out = self.u_high if self.u_high is not None else self.problem.u
        error_estimate = self.etah_vec[-1]
        self.u_high = None
        return u_out, error_estimate

    def step(self, it=None):
        """Compute one SOLVE→ESTIMATE→MARK→REFINE step.

        Returns
        -------
        tuple[Function, float]
            ``(u_out, eta_h)`` — the current solution and error estimate.
            Only returned when refinement was performed (not when
            :class:`StopIteration` is raised).
        """
        if it is None:
            V = self.problem.u.function_space()
            _, it = get_level(V.mesh())
        super().step(it)
        # Only reached if no StopIteration was raised (i.e. refinement happened)
        u_out = self.u_high if self.u_high is not None else self.problem.u
        eta_h = self.etah_vec[-1]
        return u_out, eta_h

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
        """
        Solve the primal problem or problems (i.e. with higher polynomial degree)
        and return the estimate for the error in the primal solution.
        """
        F = self.problem.F
        u = self.problem.u
        bcs = self.problem.bcs
        V = self.problem.u.function_space()
        self.Ndofs_vec.append(V.dim())

        def solve_uh():
            self.print(f'Solving primal (degree: {V.ufl_element().degree()}, dofs: {V.dim()}) ...')
            solver = NonlinearVariationalSolver(self.problem, solver_parameters=self.sp_primal,
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
            v_high = TestFunction(V_high)
            F_high = replace(F, {v_old: v_high, u: u_high})
            bcs_high = [bc.reconstruct(V=V_high, indices=bc._indices) for bc in bcs]
            problem_high = NonlinearVariationalProblem(F_high, u_high, bcs_high)

            self.print(f"Solving primal with higher order for error estimate (degree: {high_degree}, dofs: {V_high.dim()}) ...")
            solver = NonlinearVariationalSolver(problem_high, solver_parameters=self.sp_primal,
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
        """
        Solve the dual problem or problems (i.e. with higher polynomial degree)
        and return the low-order dual solution and the estimate for the error in
        the dual solution.
        """

        def solve_zh(z):
            bcs = self.problem.bcs
            J = self.goal_functional
            F = self.problem.F
            u = self.problem.u

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
            solver = LinearVariationalSolver(problem, solver_parameters=self.sp_dual, **self.dual_solver_kwargs)
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
        """Compute error estimate"""
        from firedrake.assemble import assemble
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
        """Compute cell and facet residuals R_cell, R_facet"""
        from firedrake.assemble import assemble
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
        for marker in markers.subfunctions:
            mesh = marker.function_space().mesh()
            new_mesh = mesh.refine_marked_elements(marker)
            amh, _ = get_level(mesh)
            amh.add_mesh(new_mesh)
        # Reconstruct MeshSequence with the refined meshes
        mesh = self.problem.u.function_space().mesh()
        if len(mesh) > 1:
            new_mesh = type(mesh)([get_level(m)[0][-1] for m in mesh])
            amh, _ = get_level(mesh)
            amh.add_mesh(new_mesh)

        coef_map = {}
        self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)
        self.goal_functional = refine(self.goal_functional, refine, coefficient_mapping=coef_map)
        if self.u_exact is not None:
            self.u_exact = refine(self.u_exact, refine, coefficient_mapping=coef_map)

    def write_solution(self, it):
        ws = self.options.write_solution
        if ws is False:
            return
        elif ws is True:
            should_write = True
        elif isinstance(ws, Integral):
            should_write = (it % ws == 0)
        else:
            raise ValueError(f"write_solution must be False, True, or a positive integer, got {ws!r}")
        if should_write:
            output_dir = self.options.output_dir
            run_name = self.options.run_name
            prefix = f"{output_dir}/{run_name}/{run_name}"
            self.print("Writing (primal) solution ...")
            VTKFile(f"{prefix}_solution_{it}.pvd").write(*self.problem.u.subfunctions)
            self.print("Writing (dual) solution ...")
            VTKFile(f"{prefix}_dual_solution_{it}.pvd").write(*self.z.subfunctions)


class GoalAdaptiveEigensolver(GoalAdaptiveSolverBase):
    """Solves an eigenvalue problem adaptively to minimise the error in a
    target eigenvalue.  The goal functional is :math:`J = \\lambda`.

    At each iteration the solver:

    1. Solves the primal (and, if non-self-adjoint, dual) eigenproblem at both
       degree ``p`` and ``p + dual_extra_degree``.
    2. Estimates the error in the eigenvalue via the dual-weighted residual
       formula of Larson & Bengzon.
    3. Computes cell-wise indicators using the same bubble/cone projection as
       :class:`GoalAdaptiveNonlinearVariationalSolver`.
    4. Refines the mesh by Dörfler marking.

    Parameters
    ----------
    problem
        A :class:`~.LinearEigenproblem` on the initial mesh.
    target
        Target eigenvalue used by SLEPc for spectral targeting
        (``eps_target``).
    tolerance
        Terminate when ``|eta_h| < tolerance``.
    goal_adaptive_options
        Dictionary passed to :class:`GoalAdaptiveEigenOptions`.
        Key extra entries: ``"self_adjoint"`` (bool) and ``"nev"`` (int).
    solver_parameters
        SLEPc solver parameters forwarded to :class:`~.LinearEigensolver`.
        Do not set ``eps_target`` here; it is set from ``target``.
    exact_eigenvalue
        Exact eigenvalue, if known, for computing efficiency indices.
    """

    def __init__(self,
                 problem,
                 target: float,
                 tolerance: float,
                 goal_adaptive_options: dict | None = None,
                 solver_parameters: dict | None = None,
                 exact_eigenvalue: float | None = None,
                 ):
        super().__init__(tolerance, goal_adaptive_options, exact_goal=exact_eigenvalue)
        self.problem = problem
        self.target = target
        self.sp = solver_parameters or {}

        # Set up AdaptiveMeshHierarchy
        mesh = problem.output_space.mesh()
        mh, level = get_level(mesh)
        if mh is None:
            AdaptiveMeshHierarchy(mesh)
        else:
            amh = AdaptiveMeshHierarchy(mh[0])
            for m in mh[1:level+1]:
                amh.add_mesh(m)

        self.atm = AdaptiveTransferManager()
        self._lam_h = None

    def _make_options(self, d):
        return GoalAdaptiveEigenOptions(**d)

    def solve(self):
        """Run the adaptive loop and return the eigenvalue and error estimate.

        Returns
        -------
        tuple[float, float]
            ``(lam_h, error_estimate)`` on the finest mesh.
        """
        super().solve()
        return self._lam_h, self.etah_vec[-1]

    def solve_and_estimate(self):
        """Solve the eigenproblem at two polynomial degrees, match eigenfunctions,
        and compute a global error estimate for the eigenvalue."""
        opts = self.options
        problem = self.problem
        V = problem.output_space
        self.Ndofs_vec.append(V.dim())
        self.print(f"Solving eigenproblem (degree: {V.ufl_element().degree()}, dofs: {V.dim()}) ...")

        sp_target = dict(self.sp)
        if opts.self_adjoint:
            sp_target.setdefault("eps_gen_hermitian", None)
        sp_target["eps_target"] = self.target

        # Solve at degree p
        lams, vecs = _solve_eigs(problem, opts.nev, sp_target)
        self._lam_h = lams[0]
        self._u_h = vecs[0]
        self.print(f'{"Computed eigenvalue:":40s}{self._lam_h:15.12f}')

        # Solve at degree p + dual_extra_degree (enriched primal)
        high_problem = _reconstruct_eig_degree(problem, opts.dual_extra_degree)
        self.print(f"Solving enriched eigenproblem (dofs: {high_problem.output_space.dim()}) ...")
        lams_p, vecs_p = _solve_eigs(high_problem, opts.nev, sp_target)
        self._lam_p, self._u_p = match_best(self._u_h, vecs_p, lams_p)

        if opts.self_adjoint:
            self._z_h = self._u_h
            self._z_p = self._u_p
        else:
            # Adjoint eigenproblem at degree p
            adj_problem = _make_adjoint_eig_problem(problem)
            self.print(f"Solving adjoint eigenproblem (dofs: {adj_problem.output_space.dim()}) ...")
            lamz, zs = _solve_eigs(adj_problem, opts.nev, sp_target)
            _, self._z_h = match_best(self._u_h, zs, lamz)

            # Adjoint at degree p + dual_extra_degree
            adj_high = _reconstruct_eig_degree(adj_problem, opts.dual_extra_degree)
            self.print(f"Solving enriched adjoint eigenproblem (dofs: {adj_high.output_space.dim()}) ...")
            lamzp, zsp = _solve_eigs(adj_high, opts.nev, sp_target)
            _, self._z_p = match_best(self._u_h, zsp, lamzp)

        # Dual error representative (UFL expression, may span two spaces)
        self._z_err = self._z_p - self._z_h

        eta_h, eta = self._estimate_eigenvalue_error()
        return eta_h, eta

    def _estimate_eigenvalue_error(self):
        """Compute global error estimate for the eigenvalue (Larson–Bengzon formula)."""
        from firedrake.assemble import assemble
        u_h, u_p = self._u_h, self._u_p
        z_h, z_p = self._z_h, self._z_p
        lam_h = self._lam_h
        A = self.problem._original_A
        M = self.problem._original_M

        # Low-order primal representative in the base space
        phi_h = Function(u_h.function_space())
        phi_h.interpolate(u_p)
        e = u_p - phi_h     # primal enrichment error (UFL expression)
        e_sigma = u_p - u_h

        if self.options.self_adjoint:
            sigma_h = 0.5 * float(assemble(inner(e_sigma, e_sigma) * dx))
            rhs = float(assemble(replace_both_args(A, u_h, e)
                                 - lam_h * replace_both_args(M, u_h, e)))
        else:
            e_adj = z_p - z_h
            sigma_h = 0.5 * float(assemble(inner(e_sigma, e_adj) * dx))
            rhs = 0.5 * (
                float(assemble(replace_both_args(A, u_h, e_adj)))
                - lam_h * float(assemble(replace_both_args(M, u_h, e_adj)))
                + float(assemble(replace_both_args(adjoint(A), z_h, e)))
                - lam_h * float(assemble(replace_both_args(M, z_h, e)))
            )

        denom = 1.0 - sigma_h
        eta_h = abs(rhs / denom) if abs(denom) > 1e-14 else float("nan")
        self.etah_vec.append(eta_h)
        self.print(f'{"Predicted error:":40s}{eta_h: 15.12e}')

        eta = None
        if self.goal_exact is not None:
            eta = abs(self.goal_exact - lam_h)
            self.eta_vec.append(eta)
            self.print(f'{"Exact eigenvalue:":40s}{self.goal_exact: 15.12f}')
            self.print(f'{"True error:":40s}{eta: 15.12e}')

        return eta_h, eta

    def compute_error_indicators(self):
        """Compute cell/facet residual indicators for the eigenvalue problem."""
        A = self.problem._original_A
        M = self.problem._original_M
        u_h = self._u_h
        lam_h = self._lam_h
        z_err = self._z_err
        # Residual linear form: A(u_h, v) - lam_h * M(u_h, v)
        _, trial = A.arguments()
        F_eig = replace(A, {trial: u_h}) - lam_h * replace(M, {trial: u_h})
        return _compute_residual_indicators(F_eig, z_err, self.options)

    def refine_problem(self, markers):
        """Refine the mesh and reconstruct the :class:`~.LinearEigenproblem`."""
        mesh = markers.function_space().mesh()
        new_mesh = mesh.refine_marked_elements(markers)
        amh, _ = get_level(mesh)
        amh.add_mesh(new_mesh)
        coef_map = {}
        self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)

    def write_solution(self, it):
        ws = self.options.write_solution
        if ws is False:
            return
        elif ws is True:
            should_write = True
        elif isinstance(ws, Integral):
            should_write = (it % ws == 0)
        else:
            raise ValueError(f"write_solution must be False, True, or a positive integer, got {ws!r}")
        if should_write:
            output_dir = self.options.output_dir
            run_name = self.options.run_name
            self.print("Writing (primal) eigenfunction ...")
            VTKFile(f"{output_dir}/{run_name}/{run_name}_eigenfunction_{it}.pvd"
                    ).write(*self._u_h.subfunctions)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def residual(F, test):
    """Replace the test function argument of a linear Form."""
    v, = F.arguments()
    return replace(F, {v: test})


def both(u):
    """Add u on both sides of a facet."""
    return u("+") + u("-")


def as_mixed(exprs):
    """Flatten a list of ufl.Expr objects into a vector."""
    return as_vector([e[idx] for e in exprs for idx in np.ndindex(e.ufl_shape)])


def reconstruct_degree(V, degree):
    """Reconstruct a FunctionSpace with a different polynomial degree."""
    return V.reconstruct(element=PMGPC.reconstruct_degree(V.ufl_element(), degree))


def reconstruct_bcs(bcs, V):
    """Reconstruct a list of BCs on a different FunctionSpace."""
    return [bc.reconstruct(V=V, indices=bc._indices) for bc in bcs]


def replace_both_args(bilinear_form, trial_coeff, test_coeff):
    """Substitute both arguments of a bilinear form with coefficients.

    Parameters
    ----------
    bilinear_form
        A UFL bilinear form with two arguments.
    trial_coeff
        Coefficient to substitute for the trial (second) argument.
    test_coeff
        Coefficient to substitute for the test (first) argument.

    Returns
    -------
    ufl.Form
        A 0-form (scalar integral).
    """
    test, trial = bilinear_form.arguments()
    return replace(bilinear_form, {test: test_coeff, trial: trial_coeff})


def l2_normalize(f):
    """L2-normalise a :class:`~.Function` in place and return it."""
    from firedrake.assemble import assemble
    nrm = float(assemble(inner(f, f) * dx)) ** 0.5
    if nrm > 0:
        f.assign(f / nrm)
    return f


def match_best(target, candidates, lambdas=None):
    """Return the candidate best correlated with ``target`` in L2.

    Parameters
    ----------
    target
        Reference :class:`~.Function`.
    candidates
        List of candidate :class:`~.Function` objects.
    lambdas
        List of associated eigenvalues (optional).

    Returns
    -------
    tuple
        ``(lam, aligned)`` where ``lam`` is the eigenvalue of the best match
        (or its index if ``lambdas`` is ``None``) and ``aligned`` is a
        phase/sign-aligned copy of the best candidate in the same space.
    """
    from firedrake.assemble import assemble
    nt = float(assemble(inner(target, target) * dx)) ** 0.5
    scores = []
    for i, w in enumerate(candidates):
        nw = float(assemble(inner(w, w) * dx)) ** 0.5
        if nw == 0.0:
            continue
        c = complex(assemble(inner(target, w) * dx))
        scores.append((abs(c) / (nt * nw), i, c))

    if not scores:
        raise RuntimeError("No nonzero candidate found in match_best.")

    scores.sort(key=lambda t: t[0], reverse=True)
    _, best_i, best_c = scores[0]

    aligned = candidates[best_i].copy(deepcopy=True)
    if best_c != 0:
        phase = best_c.conjugate() / abs(best_c)
        aligned.assign(phase * aligned)

    lam = lambdas[best_i] if lambdas is not None else best_i
    return lam, aligned


def _solve_eigs(problem, nev, solver_parameters):
    """Solve a :class:`~.LinearEigenproblem` and return L2-normalised eigenpairs.

    Parameters
    ----------
    problem
        :class:`~.LinearEigenproblem` to solve.
    nev
        Number of eigenvalue/eigenvector pairs to request.
    solver_parameters
        Dictionary of SLEPc solver parameters.

    Returns
    -------
    tuple[list, list]
        ``(eigenvalues, eigenfunctions)`` — both lists have length
        ``min(nconv, nev)``.
    """
    from firedrake import LinearEigensolver
    es = LinearEigensolver(problem, n_evals=nev, solver_parameters=solver_parameters)
    nconv = es.solve()
    lams, vecs = [], []
    for i in range(min(nconv, nev)):
        lams.append(es.eigenvalue(i))
        vr, _ = es.eigenfunction(i)
        vecs.append(l2_normalize(vr))
    return lams, vecs


def _reconstruct_eig_degree(problem, extra_degree):
    """Return a new :class:`~.LinearEigenproblem` on a space of degree+extra_degree.

    Uses the original unrestricted forms stored on ``problem._original_A/M/bcs``
    to avoid double-restricting on the reconstructed problem.
    """
    from firedrake import LinearEigenproblem
    A = problem._original_A
    M = problem._original_M
    bcs = problem._original_bcs
    v, u = A.arguments()
    V = u.function_space()
    high_degree = V.ufl_element().degree() + extra_degree
    V_high = reconstruct_degree(V, high_degree)
    u_high = TrialFunction(V_high)
    v_high = TestFunction(V_high)
    A_high = replace(A, {v: v_high, u: u_high})
    M_high = replace(M, {M.arguments()[0]: v_high, M.arguments()[1]: u_high})
    bcs_high = [bc.reconstruct(V=V_high, indices=bc._indices) for bc in bcs]
    return LinearEigenproblem(A_high, M_high, bcs=bcs_high,
                              bc_shift=problem.bc_shift, restrict=problem.restrict)


def _make_adjoint_eig_problem(problem):
    """Return the adjoint of a :class:`~.LinearEigenproblem`.

    For a self-adjoint problem this is identical to the primal.
    """
    from firedrake import LinearEigenproblem
    A_adj = adjoint(problem._original_A)
    return LinearEigenproblem(A_adj, problem._original_M,
                              bcs=problem._original_bcs,
                              bc_shift=problem.bc_shift, restrict=problem.restrict)


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
        :class:`GoalAdaptiveOptions` (or subclass) instance for degree
        parameters and solver parameters.

    Returns
    -------
    Function
        DG0 Function of absolute-value cell indicators.
    """
    from firedrake.assemble import assemble

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
