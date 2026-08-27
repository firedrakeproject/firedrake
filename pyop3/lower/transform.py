import textwrap

import loopy as lp
import functools

from immutabledict import immutabledict as idict

from pyop3.insn.base import (
    AbstractAssignment,
    Exscan,
    InstructionList,
    Loop,
    NullInstruction,
    StandaloneCalledFunction
)

def with_likwid_markers(knl):
    """
    See https://github.com/RRZE-HPC/likwid/wiki/TutorialMarkerC
    """
    import pylikwid

    marker_name = knl.name
    pylikwid.markerregisterregion(marker_name)

    preambles = [("99_likwid", "#include <likwid-marker.h>")]
    start_insn = lp.CInstruction((), f"LIKWID_MARKER_START(\"{marker_name}\");", id="likwid_start")
    stop_insn = lp.CInstruction((), f"LIKWID_MARKER_STOP(\"{marker_name}\");", id="likwid_stop")

    return _with_region_markers(knl, start_insn, stop_insn, preambles)


def with_petsc_event(knl):
    event_name = knl.name


    preambles = [
        (
            "99_petsc",
            textwrap.dedent(f"""
                #include <petsclog.h>

                // Prepare a dummy event so that things compile. This is overwridden using
                // the object file.
                PetscLogEvent id_{event_name} = -1;
            """)
        )
    ]

    start_insn = lp.CInstruction((), f"PetscLogEventBegin(id_{event_name}, 0, 0, 0, 0);", id="petsc_log_begin")
    stop_insn = lp.CInstruction((), f"PetscLogEventEnd(id_{event_name}, 0, 0, 0, 0);", id="petsc_log_end")

    return _with_region_markers(knl, start_insn, stop_insn, preambles)


def _with_region_markers(knl, start_insn, stop_insn, preambles):
    preambles = knl.preambles + tuple(preambles)

    assert start_insn.id is not None
    insns = (
        start_insn,
        *(insn.copy(depends_on=insn.depends_on | {start_insn.id}) for insn in knl.instructions),
        stop_insn.copy(depends_on=frozenset(insn.id for insn in knl.instructions)),
    )

    return knl.copy(preambles=preambles, instructions=insns)


def with_attach_debugger(kernel):
    debug_insn = lp.CInstruction((), "PetscAttachDebugger();", id="attach_debugger")
    insns = (
        debug_insn,
        *(insn.copy(depends_on=insn.depends_on | {debug_insn.id}) for insn in kernel.instructions),
    )
    return kernel.copy(instructions=insns)

# NOTE: Make this overloaded function into class in transform.py
# Only issue may be loopy-specific standalone_function overloading.
@functools.singledispatch
def _collect_temporary_shapes(expr):
    raise TypeError(f"No handler defined for {type(expr).__name__}")

@_collect_temporary_shapes.register(InstructionList)
def _(insn_list):
    return utils.merge_dicts(_collect_temporary_shapes(insn) for insn in insn_list)

@_collect_temporary_shapes.register(Loop)
def _(loop):
    shapes = {}
    for stmt in loop.statements:
        for temp, shape in _collect_temporary_shapes(stmt).items():
            if shape is None:
                continue
            if temp in shapes:
                assert shapes[temp] == shape
            else:
                shapes[temp] = shape
    return shapes

@_collect_temporary_shapes.register(AbstractAssignment)
@_collect_temporary_shapes.register(NullInstruction)
@_collect_temporary_shapes.register(Exscan)
def _(assignment: AbstractAssignment, /) -> idict:
    return idict()

@_collect_temporary_shapes.register
def _(call: StandaloneCalledFunction):
    return idict(
        {
            arg.buffer: lp_arg.shape
            for lp_arg, arg in zip(
                call.function.code.default_entrypoint.args, call.arguments, strict=True
            )
            if isinstance(lp_arg, lp.ArrayArg)
        }
    )
