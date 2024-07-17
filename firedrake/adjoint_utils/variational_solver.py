import copy
from functools import wraps
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from firedrake.adjoint_utils.blocks import NonlinearVariationalSolveBlock
from ufl import replace


class NonlinearVariationalProblemMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            from firedrake import derivative, adjoint, TrialFunction
            init(self, *args, **kwargs)
            self._ad_F = self.F
            self._ad_u = self.u_restrict
            self._ad_bcs = self.bcs
            self._ad_J = self.J
            try:
                # Some forms (e.g. SLATE tensors) are not currently
                # differentiable.
                dFdu = derivative(self.F,
                                  self.u_restrict,
                                  TrialFunction(self.u_restrict.function_space()))
                self._ad_adj_F = adjoint(dFdu)
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
            self._ad_nlvs = None
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

            annotate = annotate_tape(kwargs)
            if annotate:
                tape = get_working_tape()
                problem = self._ad_problem
                sb_kwargs = NonlinearVariationalSolveBlock.pop_kwargs(kwargs)
                sb_kwargs.update(kwargs)

                block = NonlinearVariationalSolveBlock(problem._ad_F == 0,
                                                       problem._ad_u,
                                                       problem._ad_bcs,
                                                       problem._ad_adj_F,
                                                       adj_cache=self._ad_adj_cache,
                                                       problem_J=problem._ad_J,
                                                       solver_params=self.parameters,
                                                       solver_kwargs=self._ad_kwargs,
                                                       ad_block_tag=self.ad_block_tag,
                                                       **sb_kwargs)
                if not self._ad_nlvs:
                    self._ad_nlvs = type(self)(
                        self._ad_problem_clone(self._ad_problem, block.get_dependencies()),
                        **self._ad_kwargs
                    )

                block._ad_nlvs = self._ad_nlvs
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
        from firedrake import Function, NonlinearVariationalProblem
        F_replace_map = {}
        J_replace_map = {}

        F_coefficients = problem.F.coefficients()
        J_coefficients = problem.J.coefficients()

        _ad_count_map = {}
        for block_variable in dependencies:
            coeff = block_variable.output
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

        nlvp = NonlinearVariationalProblem(replace(problem.F, F_replace_map),
                                           F_replace_map[problem.u_restrict],
                                           bcs=problem.bcs,
                                           J=replace(problem.J, J_replace_map))
        nlvp._constant_jacobian = problem._constant_jacobian
        nlvp._ad_count_map_update(_ad_count_map)
        return nlvp
