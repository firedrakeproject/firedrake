from functools import wraps
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape
from firedrake.adjoint_utils.blocks import ProjectBlock, SupermeshProjectBlock
from firedrake import function
from ufl.domain import extract_unique_domain


def _project_block(source, target, output, bcs, ad_block_tag, sb_kwargs,
                   project_kwargs):
    """Create the tape block recording a projection.

    Mirrors the dispatch in :func:`firedrake.projection.Projector` so that
    the tape records what the forward projection actually does, including
    the solver parameters the projection solves with. The solver parameters
    are resolved through the same code path as the Projector, so the
    projection defaults (rather than the global firedrake solver defaults)
    are what replay and adjoint solves use.
    """
    from firedrake.projection import resolve_projection_solver_parameters

    V = target.function_space() if isinstance(target, function.Function) else target
    solver_parameters = project_kwargs.get("solver_parameters")
    form_compiler_parameters = project_kwargs.get("form_compiler_parameters")
    if isinstance(source, function.Function) and extract_unique_domain(source) != V.mesh():
        return SupermeshProjectBlock(
            source, V, output, bcs, ad_block_tag=ad_block_tag,
            solver_parameters=solver_parameters,
            form_compiler_parameters=form_compiler_parameters,
            **sb_kwargs)

    solver_parameters = resolve_projection_solver_parameters(solver_parameters)
    forward_kwargs = {"solver_parameters": solver_parameters}
    if form_compiler_parameters is not None:
        forward_kwargs["form_compiler_parameters"] = form_compiler_parameters
    sb_kwargs = dict(sb_kwargs)
    sb_kwargs.setdefault("forward_kwargs", forward_kwargs)
    # The mass matrix is self-adjoint, so the adjoint solve reuses the
    # forward solver parameters. The adjoint solve takes an assembled
    # matrix, which accepts no form compiler parameters.
    sb_kwargs.setdefault("adj_kwargs", {"solver_parameters": solver_parameters})
    return ProjectBlock(source, V, output, bcs, ad_block_tag=ad_block_tag,
                        solver_parameters=solver_parameters, **sb_kwargs)


def annotate_project(project):
    @wraps(project)
    def wrapper(*args, **kwargs):
        """The project call performs an equation solve, and so it too must be annotated so that the
        adjoint and tangent linear models may be constructed automatically by pyadjoint.

        To disable the annotation of this function, just pass :py:data:`annotate=False`. This is useful in
        cases where the solve is known to be irrelevant or diagnostic for the purposes of the adjoint
        computation (such as projecting fields to other function spaces for the purposes of
        visualisation)."""

        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs)
        block = None
        if annotate:
            bcs = kwargs.get("bcs", [])
            sb_kwargs = ProjectBlock.pop_kwargs(kwargs)
            if isinstance(args[1], function.Function):
                # block should be created before project because output might also be an input that needs checkpointing
                block = _project_block(args[0], args[1], args[1], bcs,
                                       ad_block_tag, sb_kwargs, kwargs)

        with stop_annotating():
            output = project(*args, **kwargs)

        if annotate:
            if block is None:
                block = _project_block(args[0], args[1], output, bcs,
                                       ad_block_tag, sb_kwargs, kwargs)
            tape = get_working_tape()
            tape.add_block(block)
            block.add_output(output.create_block_variable())

        return output

    return wrapper
