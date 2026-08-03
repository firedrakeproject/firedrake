import firedrake
from functools import wraps
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape
from firedrake.adjoint_utils.blocks import SupermeshProjectBlock
from firedrake.adjoint_utils.variational_solver import NonlinearVariationalSolverMixin


def annotate_project(project):
    """
    A wrapper for the bare project() call to handle
    switching annotation on/off without dispatching
    into all the types of projectors.
    """

    @wraps(project)
    def wrapper(*args, **kwargs):
        annotate = annotate_tape(kwargs)
        if annotate:
            return project(*args, **kwargs)

        with stop_annotating():
            return project(*args, **kwargs)

    return wrapper


def annotate_super_project(project):
    """
    A wrapper for SupermeshProjector.project(), only handles
    the code path that leads to the creation of a
    SupermeshProjectBlock.
    """

    @wraps(project)
    def wrapper(self, **kwargs):
        annotate = annotate_tape(kwargs)
        V = self.target.function_space()
        if annotate:
            bcs = kwargs.get("bcs", [])
            sb_kwargs = SupermeshProjectBlock.pop_kwargs(kwargs)
            if self._target_is_function:
                # block should be created before project because output might also be an input that needs checkpointing
                block = SupermeshProjectBlock(self.source, V, self.target, bcs, ad_block_tag=self.ad_block_tag, **sb_kwargs)

        with stop_annotating():
            output = project(self, **kwargs)

        if annotate:
            tape = get_working_tape()
            if not self._target_is_function:
                block = SupermeshProjectBlock(self.source, V, output, bcs, ad_block_tag=self.ad_block_tag, **sb_kwargs)
            tape.add_block(block)
            block.add_output(output.create_block_variable())

        return output

    return wrapper


class ProjectorMixin(NonlinearVariationalSolverMixin):
    """A wrapper over the nonlinear variational solver mixin for projectors.

    This establishes the form used for adjoint evaluation of the projector.
    """

    def __init__(self, ad_block_tag=None):
        V = self.target.function_space()
        mesh = V.mesh()
        ad_target = firedrake.Function(V)

        dx = firedrake.dx(mesh)
        w = firedrake.TestFunction(V)
        Pv = firedrake.TrialFunction(V)
        a = firedrake.inner(Pv, w) * dx
        L = firedrake.inner(self.source, w) * dx
        p = firedrake.LinearVariationalProblem(a, L, ad_target)

        # If we project from an expression containing u (e.g. u + c) onto u, the
        # resulting form looks nonlinear. However, the projection is in effect a
        # solve onto a temporary function (here ad_target), followed by an implicit
        # assignment. Instead of representing this as an explicit solve and assign
        # block, we solve for a temporary function and modify _ad_u on the problem.
        # This is used for creating the output variable (and thus the dependency
        # resolution), in effect merging the implicit assignment into the block
        # output phase.
        p._ad_u = self.target

        solver_params = firedrake.LinearVariationalSolver.DEFAULT_SNES_PARAMETERS | self.solver_parameters

        self._init_as_solver(p, forward_kwargs={"solver_parameters": solver_params}, ad_block_tag=ad_block_tag)

    @NonlinearVariationalSolverMixin._ad_annotate_init
    def _init_as_solver(self, problem, **kwargs):
        return

    @staticmethod
    def _ad_annotate_project(project):
        return NonlinearVariationalSolverMixin._ad_annotate_solve(project)
