import copy
from functools import wraps
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from firedrake.adjoint_utils.blocks import NonlinearVariationalSolveBlock
from firedrake.ufl_expr import derivative, adjoint
from ufl import replace


class NonlinearVariationalProblemMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self._ad_F = self.F
            self._ad_u = self.u_restrict
            self._ad_bcs = self.bcs
            self._ad_J = self.J
            try:
                # Some forms (e.g. SLATE tensors) are not currently
                # differentiable.
                dFdu = derivative(self.F, self.u_restrict)
                try:
                    self._ad_adj_F = adjoint(dFdu)
                except ValueError:
                    # Try again without expanding derivatives,
                    # as dFdu might have been simplied to an empty Form
                    self._ad_adj_F = adjoint(dFdu, derivatives_expanded=True)
            except (TypeError, NotImplementedError):
                self._ad_adj_F = None
            self._ad_kwargs = {'Jp': self.Jp, 'form_compiler_parameters': self.form_compiler_parameters, 'is_linear': self.is_linear}
            self._ad_count_map = {}
        return wrapper

    def _ad_count_map_update(self, updated_ad_count_map):
        self._ad_count_map = updated_ad_count_map


class NonlinearVariationalSolverMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, problem, *args, **kwargs):
            self.ad_block_tag = kwargs.pop("ad_block_tag", None)
            init(self, problem, *args, **kwargs)
            self._ad_problem = problem
            self._ad_args = args
            self._ad_kwargs = kwargs
            self._ad_solvers = {"forward_nlvs": None, "adjoint_lvs": None,
                                "recompute_count": 0}
            self._ad_adj_cache = {}

        return wrapper

    @staticmethod
    def _ad_annotate_solve(solve):
        @wraps(solve)
        def wrapper(self, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
            Firedrake solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
            for the purposes of the adjoint computation (such as projecting fields to other function spaces
            for the purposes of visualisation)."""
            from firedrake import LinearVariationalSolver
            annotate = annotate_tape(kwargs)
            if annotate:
                tape = get_working_tape()
                problem = self._ad_problem
                sb_kwargs = NonlinearVariationalSolveBlock.pop_kwargs(kwargs)
                sb_kwargs.update(kwargs)

                block = NonlinearVariationalSolveBlock(problem._ad_F == 0,
                                                       problem._ad_u,
                                                       problem._ad_bcs,
                                                       adj_cache=self._ad_adj_cache,
                                                       problem_J=problem._ad_J,
                                                       solver_params=self.parameters,
                                                       solver_kwargs=self._ad_kwargs,
                                                       ad_block_tag=self.ad_block_tag,
                                                       **sb_kwargs)

                # Forward variational solver.
                if not self._ad_solvers["forward_nlvs"]:
                    self._ad_solvers["forward_nlvs"] = type(self)(
                        self._ad_problem_clone(self._ad_problem, block.get_dependencies()),
                        **self._ad_kwargs
                    )

                # Adjoint variational solver.
                if not self._ad_solvers["adjoint_lvs"]:
                    with stop_annotating():
                        self._ad_solvers["adjoint_lvs"] = LinearVariationalSolver(
                            self._ad_adj_lvs_problem(block, problem._ad_adj_F),
                            *block.adj_args, **block.adj_kwargs)
                        if self._ad_problem._constant_jacobian:
                            self._ad_solvers["update_adjoint"] = False

                block._ad_solvers = self._ad_solvers

                tape.add_block(block)

            with stop_annotating():
                out = solve(self, **kwargs)

            if annotate:
                block.add_output(self._ad_problem._ad_u.create_block_variable())

            return out

        return wrapper

    @no_annotations
    def _ad_problem_clone(self, problem, dependencies):
        """Replaces every coefficient in the residual and jacobian with a deepcopy to return
        a clone of the original NonlinearVariationalProblem instance. We'll be modifying the
        numerical values of the coefficients in the residual and jacobian, so in order not to
        affect the user-defined self._ad_problem.F, self._ad_problem.J and self._ad_problem.u
        expressions, we'll instead create clones of them.
        """
        from firedrake import NonlinearVariationalProblem
        _ad_count_map, J_replace_map, F_replace_map = self._build_count_map(
            problem.J, dependencies, F=problem.F)
        nlvp = NonlinearVariationalProblem(replace(problem.F, F_replace_map),
                                           F_replace_map[problem.u_restrict],
                                           bcs=problem.bcs,
                                           J=replace(problem.J, J_replace_map))
        nlvp.is_linear = problem.is_linear
        nlvp._constant_jacobian = problem._constant_jacobian
        nlvp._ad_count_map_update(_ad_count_map)
        return nlvp

    @no_annotations
    def _ad_adj_lvs_problem(self, block, adj_F):
        """Create the adjoint variational problem."""
        from firedrake import Function, Cofunction, LinearVariationalProblem
        # Homogeneous boundary conditions for the adjoint problem
        # when Dirichlet boundary conditions are applied.
        bcs = block._homogenize_bcs()
        adj_sol = Function(block.function_space)
        right_hand_side = Cofunction(block.function_space.dual())
        tmp_problem = LinearVariationalProblem(
            adj_F, right_hand_side, adj_sol, bcs=bcs,
            constant_jacobian=self._ad_problem._constant_jacobian)
        # The `block.adj_F` coefficients hold the output references.
        # We do not want to modify the user-defined values. Hence, the adjoint
        # linear variational problem is created with a deep copy of the
        # `block.adj_F` coefficients.
        _ad_count_map, J_replace_map, _ = self._build_count_map(
            adj_F, block._dependencies)
        lvp = LinearVariationalProblem(
            replace(tmp_problem.J, J_replace_map), right_hand_side, adj_sol,
            bcs=tmp_problem.bcs,
            constant_jacobian=self._ad_problem._constant_jacobian)
        lvp._ad_count_map_update(_ad_count_map)
        return lvp

    def _build_count_map(self, J, dependencies, F=None):
        from firedrake import Function

        F_replace_map = {}
        J_replace_map = {}
        if F:
            F_coefficients = F.coefficients()
        J_coefficients = J.coefficients()

        _ad_count_map = {}
        for block_variable in dependencies:
            coeff = block_variable.output
            if F:
                if coeff in F_coefficients and coeff not in F_replace_map:
                    if isinstance(coeff, Function) and coeff.ufl_element().family() == "Real":
                        F_replace_map[coeff] = copy.deepcopy(coeff)
                    else:
                        F_replace_map[coeff] = coeff.copy(deepcopy=True)
                    _ad_count_map[F_replace_map[coeff]] = coeff.count()

            if coeff in J_coefficients and coeff not in J_replace_map:
                if coeff in F_replace_map:
                    J_replace_map[coeff] = F_replace_map[coeff]
                elif isinstance(coeff, Function) and coeff.ufl_element().family() == "Real":
                    J_replace_map[coeff] = copy.deepcopy(coeff)
                else:
                    J_replace_map[coeff] = coeff.copy()
                _ad_count_map[J_replace_map[coeff]] = coeff.count()
        return _ad_count_map, J_replace_map, F_replace_map
