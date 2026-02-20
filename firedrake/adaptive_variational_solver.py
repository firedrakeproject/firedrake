from firedrake.petsc import PETSc
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.ufl_expr import TestFunction, TrialFunction, derivative, action, adjoint
from firedrake.variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                          LinearVariationalProblem, LinearVariationalSolver)
from firedrake.solving import solve
from firedrake.norms import norm

from firedrake.preconditioners.pmg import PMGPC
from firedrake.mg import utils
from firedrake.mg.ufl_utils import coarsen, CoarseningError
from firedrake.mg.adaptive_hierarchy import AdaptiveMeshHierarchy
from firedrake.mg.adaptive_transfer_manager import AdaptiveTransferManager
from firedrake.output import VTKFile

from finat.ufl import BrokenElement, FiniteElement
from ufl import avg, dot, div, grad, dx, ds, dS, inner, jump, as_vector, replace, FacetNormal
import ufl

from functools import singledispatch
from pathlib import Path
import numpy as np
import csv


class GoalAdaptiveOptions:
    def __init__(self, config: dict):
        self.manual_indicators = config.get("manual_indicators", False)  # Used for manual indicators (only implemented for Poisson but could be overriden)
        self.dorfler_alpha = config.get("dorfler_alpha", 0.5)  # Dorfler marking parameter, default 0.5
        self.max_iterations = config.get("max_iterations", 10)
        self.output_dir = config.get("output_dir", "./output")
        self.dual_extra_degree = config.get("dual_extra_degree", 1)
        self.cell_residual_extra_degree = config.get("cell_residual_extra_degree", 0)
        self.facet_residual_extra_degree = config.get("facet_residual_extra_degree", 0)
        self.write_at_iteration = config.get("write_at_iteration", True)
        self.use_adjoint_residual = config.get("use_adjoint_residual", False)  # For switching between primal and primal + adjoint residuals
        self.exact_indicators = config.get("exact_indicators", False)  # Maybe remove
        self.uniform_refinement = config.get("uniform_refinement", False)
        self.primal_low_method = config.get("primal_low_method", "interpolate")
        self.dual_low_method = config.get("dual_low_method", "interpolate")
        self.write_mesh = config.get("write_mesh", "all")  # Default all, options: "first_and_last" "by iteration" "none"
        self.write_mesh_iteration_vector = config.get("write_iteration_vector", [])
        self.write_mesh_iteration_interval = config.get("write_iteration_interval", 1)
        self.write_solution = config.get("write_solution", "all")  # Default all, options: "first_and_last" "by iteration" "none"
        self.write_solution_iteration_vector = config.get("write_iteration_vector", [])
        self.write_solution_iteration_interval = config.get("write_solution", "all")  # Default all, options: "first_and_last" "by iteration" "none"
        self.results_file_name = config.get("results_file_name", None)
        self.nev = config.get("nev", 5)
        self.run_name = config.get("run_name", None)
        self.form_compiler_parameters = config.get("form_compiler_parameters", None)

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


class GoalAdaptiveNonlinearVariationalSolver():
    """Solves a nonlinear variational problem to minimise the error in a
    user-specified goal functional.  We do this by adaptively refining the mesh
    based on the solution to a dual problem - which links the goal functional
    to the PDE.
    """
    def __init__(self, problem: NonlinearVariationalProblem, goal_functional, tolerance: float, solver_parameters: dict,
                 *, primal_solver_parameters=None, dual_solver_parameters=None, exact_solution=None, exact_goal=None, source_term, verbose=True,
                 nullspace=None):
        # User input vars
        self.problem = problem
        self.goal_functional = goal_functional
        self.tolerance = tolerance
        self.sp_primal = primal_solver_parameters
        self.sp_dual = dual_solver_parameters
        self.u_exact = as_mixed(exact_solution) if isinstance(exact_solution, (tuple, list)) else exact_solution
        self.goal_exact = exact_goal
        self.options = GoalAdaptiveOptions(solver_parameters)
        self.source_term = source_term

        # Derived vars
        self.verbose = verbose
        V = problem.u.function_space()
        self.element = V.ufl_element()
        self.degree = self.element.degree()
        self.nullspace = nullspace
        self.atm = AdaptiveTransferManager()
        self.amh = AdaptiveMeshHierarchy(V.mesh())

        # Data storage and writing
        self.output_dir = Path(self.options.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # ensures folder exists
        self.N_vec = []
        self.Ndual_vec = []
        self.eta_vec = []
        self.etah_vec = []
        self.etaTsum_vec = []
        self.eff1_vec = []
        self.eff2_vec = []
        self.eff3_vec = []

    def solve_primal(self):
        F = self.problem.F
        u = self.problem.u
        bcs = self.problem.bcs
        V = self.problem.u.function_space()
        self.N_vec.append(V.dim())

        def solve_uh():
            self.print(f'Solving primal (degree: {self.degree}, dofs: {V.dim()}) ...')
            solver = NonlinearVariationalSolver(self.problem, solver_parameters=self.sp_primal)
            solver.set_transfer_manager(self.atm)
            solver.solve()

        solve_uh()
        if self.options.use_adjoint_residual:
            # Now solve in higher space
            high_degree = self.degree + self.options.dual_extra_degree  # Use dual degree
            high_element = PMGPC.reconstruct_degree(self.element, high_degree)
            Vhigh = V.reconstruct(element=high_element)
            u_high = Function(Vhigh).interpolate(u)
            self.print("u norm:", norm(u))
            self.print("u_high norm:", norm(u_high))

            v_old, = F.arguments()
            v_high = TestFunction(Vhigh)
            F_high = replace(F, {v_old: v_high, u: u_high})
            bcs_high = reconstruct_bcs(bcs, Vhigh)
            problem_high = NonlinearVariationalProblem(F_high, u_high, bcs_high)

            self.print(f"Solving primal in higher space for error estimate (degree: {high_degree}, dofs: {Vhigh.dim()}) ...")
            solver = NonlinearVariationalSolver(problem_high, solver_parameters=self.sp_primal)
            solver.set_transfer_manager(self.atm)
            solver.solve()

            if self.options.primal_low_method == "solve":
                solve_uh()
            elif self.options.primal_low_method == "project":
                u.project(u_high)
            else:
                # Default to interpolation - but gives bad results?
                u.interpolate(u_high)
            u_err = u_high - u
        else:
            u_err = None
        return u_err

    def solve_dual(self):
        J = self.goal_functional
        F = self.problem.F
        u = self.problem.u
        bcs = self.problem.bcs
        V = u.function_space()

        def solve_zh(z):
            Z = z.function_space()
            self.print(f"Solving dual (degree: {Z.ufl_element().degree()}, dofs: {Z.dim()}) ...")
            dF = derivative(F, u, TrialFunction(Z))
            dJ = derivative(J, u, TestFunction(Z))
            G = action(adjoint(dF), z) - dJ
            a = derivative(G, z)
            bcs_dual = [bc.reconstruct(V=Z, indices=bc._indices, g=0) for bc in bcs]
            problem = LinearVariationalProblem(a, dJ, z, bcs_dual)
            solver = LinearVariationalSolver(problem, solver_parameters=self.sp_dual)
            solver.set_transfer_manager(self.atm)
            solver.solve()

        # Higher-order dual soluton
        dual_degree = self.degree + self.options.dual_extra_degree
        dual_element = PMGPC.reconstruct_degree(self.element, dual_degree)
        Vdual = V.reconstruct(element=dual_element)
        self.z = Function(Vdual)

        self.Ndual_vec.append(Vdual.dim())
        solve_zh(self.z)

        # Lower-order dual soluton
        self.z_lo = Function(V)
        if self.options.dual_low_method == "solve":
            solve_zh(self.z_lo)
        elif self.options.dual_low_method == "project":
            self.z_lo.project(self.z)
        else:
            # Default to interpolation
            self.z_lo.interpolate(self.z)
        z_err = self.z - self.z_lo
        return z_err

    def compute_etah(self, u_err, z_err):
        """Compute error estimate"""
        from firedrake.assemble import assemble
        J = self.goal_functional
        F = self.problem.F
        u = self.problem.u

        primal_err = abs(assemble(residual(F, z_err)))
        if self.options.use_adjoint_residual:
            dF = derivative(F, u)
            dJ = derivative(J, u)
            G = action(adjoint(dF), self.z_lo) - dJ

            dual_err = abs(assemble(residual(G, u_err)))
            eta_h = 0.5 * abs(primal_err + dual_err)
        else:
            eta_h = primal_err

        self.etah_vec.append(eta_h)

        Juh = assemble(J)
        self.print(f'{"Computed goal":45s}{"J(uh):":8s}{Juh:15.12f}')
        self.Juh = Juh

        if self.goal_exact is not None:
            Ju = self.goal_exact
        elif self.u_exact is not None:
            fcp = self.options.form_compiler_parameters
            Ju = assemble(replace(J, {u: self.u_exact}), form_compiler_parameters=fcp)

        if self.u_exact is not None or self.goal_exact is not None:
            eta = abs(Juh - Ju)
            self.eta_vec.append(eta)
            self.print(f'{"Exact goal":45s}{"J(u):":8s}{Ju:15.12f}')
            self.print(f'{"True error, |J(u) - J(u_h)|":45s}{"eta:":8s}{eta:15.12f}')
        else:
            eta = None

        if self.options.use_adjoint_residual:
            self.print(f'{"Primal error, |rho(u_h;z-z_h)|:":45s}{"eta_pri:":8s}{primal_err:15.12f}')
            self.print(f'{"Dual error, |rho*(z_h;u-u_h)|:":45s}{"eta_adj:":8s}{dual_err:15.12f}')
            self.print(f'{"Difference":35s}{"|eta_pri - eta_adj|:":18s}{abs(primal_err-dual_err):19.12e}')
            self.print(f'{"Predicted error, 0.5|rho+rho*|":45s}{"eta_h:":8s}{eta_h:15.12f}')
        else:
            self.print(f'{"Predicted error, |rho(u_h;z-z_h)|":45s}{"eta_h:":8s}{eta_h:15.12f}')
        return eta_h, eta

    def automatic_error_indicators(self, u_err, z_err):
        """Compute cell and facet residuals R_cell, R_facet"""
        from firedrake.assemble import assemble
        J = self.goal_functional
        F = self.problem.F
        u = self.problem.u
        V = u.function_space()

        mesh = V.mesh().unique()
        dim = mesh.topological_dimension
        cell = mesh.ufl_cell()
        variant = "integral"
        cell_residual_degree = self.degree + self.options.cell_residual_extra_degree
        facet_residual_degree = self.degree + self.options.facet_residual_extra_degree
        # ------------------------------- Primal residual -------------------------------
        # Cell bubbles
        B = FunctionSpace(mesh, "B", dim+1, variant=variant)  # Bubble function space
        bubbles = Function(B).assign(1)  # Bubbles
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
            # r*(v) = J"(u)[v] - A"_u(u)[v, z]  since F = A(u;v) - L(v)
            dF = derivative(F, u, TrialFunction(V))
            dJ = derivative(J, u, TestFunction(V))
            rstar = action(adjoint(dF), self.z_lo) - dJ

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
            # indicators: 0.5 * (primal + dual)
            etaT = assemble(0.5*(eta_primal + eta_dual))

            with eta_primal.dat.vec as evec:
                self.eta_primal_total = abs(evec.sum())
            with eta_dual.dat.vec as evec:
                self.eta_dual_total = abs(evec.sum())
            self.print(f'{"Sum of primal refinement indicators":45s}{"Σeta_K:":8s}{self.eta_dual_total:15.12f}')
            self.print(f'{"Sum of dual refinement indicators":45s}{"Σeta_K:":8s}{self.eta_dual_total:15.12f}')
        else:
            etaT = eta_primal

        # XXX Exact error indicators (experimental - ignore)
        if self.options.exact_indicators:
            u_err_exact = self.u_exact - u
            eta_dual_exact = assemble(
                inner(inner(Rcell_star, u_err_exact), test)*dx
                + inner(avg(inner(Rfacet_star, u_err_exact)), both(test))*dS
                + inner(inner(Rfacet_star, u_err_exact), test)*ds
            )
            udiff = assemble(eta_dual_exact - eta_dual)
            with udiff.dat.vec as uvec:
                unorm = uvec.norm()
            self.print("L2 error in (dual) refinement indicators: ", unorm)
        return etaT

    def manual_error_indicators(self, u_err, z_err):
        """ Currently only implemented for Poisson, but can be overriden. To adapt to other PDEs, replace the form of
        etaT = assemble() to the symbolic form of the error indicators. This form is usually obtained by integrating
        the weak form by parts (to recover the strong form) and redistributing facet fluxes.
        """
        from firedrake.assemble import assemble
        u = self.problem.u
        mesh = u.function_space().mesh().unique()
        n = FacetNormal(mesh)
        f = self.source_term
        DG0 = FunctionSpace(mesh, "DG", degree=0)
        test = TestFunction(DG0)
        etaT = assemble(
            inner(f + div(grad(u)), z_err * test) * dx
            - inner(jump(grad(u), n), z_err * avg(test)) * dS
            - inner(dot(grad(u), n), z_err * test) * ds
        )
        with etaT.dat.vec as evec:
            evec.abs()
        return etaT

    def compute_efficiency(self, etaT, eta_h, eta):
        with etaT.dat.vec as evec:
            etaT_total = abs(evec.sum())

        self.etaTsum_vec.append(etaT_total)
        self.print(f'{"Sum of refinement indicators":45s}{"Ση_K:":8s}{etaT_total:15.12f}')

        if self.u_exact is not None or self.goal_exact is not None:
            # Compute efficiency indices
            eff1 = eta_h / eta
            eff2 = etaT_total / eta
            self.eff1_vec.append(eff1)
            self.eff2_vec.append(eff2)
            self.print(f'{"Effectivity index 1":45s}{"η_h/η:":8s}{eff1:7.4f}')
            self.print(f'{"Effectivity index 2":45s}{"Ση_K/η:":8s}{eff2:7.4f}')
        else:
            eff3 = etaT_total / eta_h
            self.eff3_vec.append(eff3)
            self.print(f'{"Effectivity index:":45s}{"Ση_K/η_h:":8s}{eff3:7.4f}')

    def set_adaptive_cell_markers(self, etaT):
        """Mark cells for refinement (Dorfler marking)"""
        mesh = self.amh[-1]
        if mesh.comm.size == 1:
            with etaT.dat.vec as evec:
                etaT_total = abs(evec.sum())
            threshold = self.options.dorfler_alpha * etaT_total
            sorted_indices = np.argsort(-etaT.dat.data_ro)
            sorted_etaT = etaT.dat.data[sorted_indices]
            cumulative_sum = np.cumsum(sorted_etaT)
            M = np.searchsorted(cumulative_sum, threshold) + 1
            marked_cells = sorted_indices[:M]
        else:
            # TODO implement a parallel sort
            with etaT.dat.vec_ro as evec:
                _, eta_max = evec.max()
            threshold = self.options.dorfler_alpha * eta_max
            marked_cells = etaT.dat.data_ro > threshold

        markers = Function(etaT.function_space())
        markers.dat.data_wo[marked_cells] = 1
        return markers

    def set_uniform_cell_markers(self):
        """Uniform marking for comparison tests"""
        mesh = self.amh[-1]
        markers_space = FunctionSpace(mesh, "DG", 0)
        markers = Function(markers_space)
        markers.assign(1)
        return markers

    def refine_problem(self, markers):
        mesh = self.amh[-1]
        new_mesh = mesh.refine_marked_elements(markers)
        self.amh.add_mesh(new_mesh)

        coef_map = {}
        self.problem = refine(self.problem, refine, coefficient_mapping=coef_map)
        self.goal_functional = refine(self.goal_functional, refine, coefficient_mapping=coef_map)
        if self.u_exact is not None:
            self.u_exact = refine(self.u_exact, refine, coefficient_mapping=coef_map)
        if self.source_term is not None:
            self.source_term = refine(self.source_term, refine, coefficient_mapping=coef_map)

    def solve(self):
        for it in range(self.options.max_iterations):
            if self.step():
                break

    def step(self):
        it = len(self.amh) - 1
        self.print(f"---------------------------- [MESH LEVEL {it}] ----------------------------")
        self.write_mesh(it)
        u_err = self.solve_primal()
        z_err = self.solve_dual()
        self.write_solution(it)
        eta_h, eta = self.compute_etah(u_err, z_err)
        if eta_h < self.tolerance:
            self.print("Error estimate below tolerance, finished.")
            return 1

        if it == self.options.max_iterations - 1:
            self.print(f"Maximum iteration ({self.options.max_iterations}) reached. Exiting.")
            return 1

        if self.options.uniform_refinement:
            self.print("Refining uniformly")
            markers = self.set_uniform_cell_markers()
        else:
            self.print("Computing local refinement indicators eta_K ...")
            if self.options.manual_indicators:
                etaT = self.manual_error_indicators(u_err, z_err)
            else:
                etaT = self.automatic_error_indicators(u_err, z_err)
            self.compute_efficiency(etaT, eta_h, eta)
            markers = self.set_adaptive_cell_markers(etaT)

        if self.options.write_at_iteration:
            self.print("Appending data ...")
            self.append_data()
        self.print("Transferring problem to new mesh ...")
        self.refine_problem(markers)
        if not self.options.write_at_iteration:
            self.print("Writing data ...")
            self.write_data()
        return 0

    # Writing functions: --------------------------------------

    def print(self, *args, **kwargs):
        if self.verbose:
            PETSc.Sys.Print(*args, **kwargs)

    def write_data(self):
        # Write to file
        if self.options.results_file_name is None:
            file_path = self.output_dir / "results.csv"
        else:
            file_path = self.output_dir / self.options.results_file_name
        rows = list(zip(self.N_vec, self.Ndual_vec, self.eta_vec, self.etah_vec, self.etaTsum_vec, self.eff1_vec, self.eff2_vec))
        headers = ("N", "Ndual", "eta", "eta_h", "sum_eta_T", "eff1", "eff2")
        with open(file_path, "w", newline="") as file:
            w = csv.writer(file)
            w.writerow(headers)
            w.writerows(rows)

    def append_data(self):
        it = len(self.amh) - 1
        Juh = self.Juh
        run_name = self.options.run_name
        if run_name is None:
            file_path = self.output_dir / "results.csv"
        else:
            file_path = self.output_dir / f"{run_name}/{run_name}_results.csv"
        if self.u_exact is None and self.goal_exact is None and not self.options.uniform_refinement:
            headers = ("iteration", "N", "Ndual", "Juh", "eta_h", "sum_eta_T")
            row = (
                it,
                self.N_vec[-1], self.Ndual_vec[-1], Juh, self.etah_vec[-1], self.etaTsum_vec[-1]
            )
        elif self.options.uniform_refinement:
            headers = ("iteration", "N", "Ndual", "Juh", "eta", "eta_h")
            row = (
                it,
                self.N_vec[-1], self.Ndual_vec[-1], Juh, self.eta_vec[-1], self.etah_vec[-1]
            )
        else:
            headers = ("iteration", "N", "Ndual", "Juh", "eta", "eta_h", "sum_eta_T", "eff1", "eff2")
            row = (
                it,
                self.N_vec[-1], self.Ndual_vec[-1], Juh, self.eta_vec[-1], self.etah_vec[-1], self.etaTsum_vec[-1], self.eff1_vec[-1], self.eff2_vec[-1]
            )

        file_path.parent.mkdir(parents=True, exist_ok=True)  # create directories if missing
        if it == 0:
            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerow(row)
        else:
            with open(file_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

    def write_solution(self, it):
        should_write = False
        write_solution = self.options.write_mesh
        if write_solution == "all":
            should_write = True
        elif write_solution == "first_and_last":
            if it == 0 or it == self.options.max_iterations:
                should_write = True
        elif write_solution == "by_iteration":
            # Case A: user gave specific iterations (list/tuple/set)
            if getattr(self.options, "write_solution_iteration_vector", None) is not None:
                # allow any iterable; convert to set for O(1) lookup
                targets = set(self.options.write_iteration_vector)
                should_write = it in targets
            # Case B: otherwise use interval (positive int)
            elif getattr(self.options, "write_solution_iteration_interval", None) is not None:
                interval = int(self.options.write_iteration_interval)
                if interval <= 0:
                    raise ValueError("write_solution_iteration_interval must be a positive integer")
                should_write = (it % interval == 0)  # includes it=0
        if should_write:
            run_name = self.options.run_name
            self.print("Writing (primal) solution ...")
            u = self.problem.u
            VTKFile(self.output_dir / f"{run_name}/{run_name}_solution_{it}.pvd").write(*u.subfunctions)
            self.print("Writing (dual) solution ...")
            VTKFile(self.output_dir / f"{run_name}/{run_name}_dual_solution_{it}.pvd").write(*self.z.subfunctions)

    def write_mesh(self, it):
        mesh = self.problem.u.function_space().mesh()
        should_write = False
        write_mesh = self.options.write_mesh
        if write_mesh == "all":
            should_write = True
        elif write_mesh == "first_and_last":
            if it == 0 or it == self.options.max_iterations:
                should_write = True
        elif write_mesh == "by_iteration":
            # Case A: user gave specific iterations (list/tuple/set)
            if getattr(self.options, "write_mesh_iteration_vector", None) is not None:
                # allow any iterable; convert to set for O(1) lookup
                targets = set(self.options.write_iteration_vector)
                should_write = it in targets
            # Case B: otherwise use interval (positive int)
            elif getattr(self.options, "write_mesh_iteration_interval", None) is not None:
                interval = int(self.options.write_iteration_interval)
                if interval <= 0:
                    raise ValueError("write_mesh_iteration_interval must be a positive integer")
                should_write = (it % interval == 0)  # includes it=0
        if should_write:
            self.print("Writing mesh ...")
            run_name = self.options.run_name
            VTKFile(self.output_dir / f"{run_name}/{run_name}_mesh_{it}.pvd").write(mesh)


# Utility functions

def residual(F, test):
    v, = F.arguments()
    return replace(F, {v: test})


def both(u):
    return u("+") + u("-")


def as_mixed(exprs):
    return as_vector([e[idx] for e in exprs for idx in np.ndindex(e.ufl_shape)])


# Should no longer be needed
def reconstruct_bc_value(bc, V):
    if not isinstance(bc._original_arg, Function):
        return bc._original_arg
    return Function(V).interpolate(bc._original_arg)


def reconstruct_bcs(bcs, V):
    """Reconstruct a list of bcs"""
    new_bcs = []
    for bc in bcs:
        V_ = V
        for index in bc._indices:
            V_ = V_.sub(index)
        g = reconstruct_bc_value(bc, V_)
        new_bcs.append(bc.reconstruct(V=V_, g=g))

    return new_bcs


@singledispatch
def refine(expr, self, coefficient_mapping=None):
    return coarsen(expr, self, coefficient_mapping=coefficient_mapping)  # fallback to original


@refine.register(ufl.Mesh)
@refine.register(ufl.MeshSequence)
def refine_mesh(mesh, self, coefficient_mapping=None):
    hierarchy, level = utils.get_level(mesh)
    if hierarchy is None:
        raise CoarseningError("No mesh hierarchy available")
    return hierarchy[level + 1]


@refine.register(Cofunction)
@refine.register(Function)
def refine_function(expr, self, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}
    new = coefficient_mapping.get(expr)
    if new is None:
        Vf = expr.function_space()
        Vc = self(Vf, self)
        new = Function(Vc, name=f"coarse_{expr.name()}")
        new.interpolate(expr)
        coefficient_mapping[expr] = new
    return new


# Only required for my examples:
def getlabels(mesh, codim):
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=codim)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
    return names_to_labels
