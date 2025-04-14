import textwrap

import loopy as lp


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


def with_breakpoint(kernel):
    debug_insn = lp.CInstruction((), "PetscAttachDebugger();", id="attach_debugger")
    insns = (
        debug_insn,
        *(insn.copy(depends_on=insn.depends_on | {debug_insn.id}) for insn in kernel.instructions),
    )
    return kernel.copy(instructions=insns)
