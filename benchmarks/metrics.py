"""Instrument generated finite element kernels."""

import os
import tempfile
import time
import types

import numpy


def isolate_caches(prefix: str) -> None:
    """Point the TSFC and PyOP2 caches at a fresh directory.

    Parameters
    ----------
    prefix
        Prefix for the temporary cache directory.

    Notes
    -----
    Must run before ``import firedrake``, which fills these variables in if
    they are unset.  A warm disk cache hides code generation, which is part
    of the quantity being measured.
    """
    cache = tempfile.mkdtemp(prefix=prefix)
    os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] = os.path.join(cache, "tsfc")
    os.environ["PYOP2_CACHE_DIR"] = os.path.join(cache, "pyop2")


def kernel_metrics(kernel: object) -> types.SimpleNamespace:
    """Separate writable intermediates from immutable tables.

    Parameters
    ----------
    kernel
        Compiled TSFC kernel.

    Returns
    -------
    types.SimpleNamespace
        ``flops``, the scalar and array temporary counts, the entries they
        hold, and the length of the generated AST.

    Notes
    -----
    Loopy represents compile-time quadrature and tabulation data as
    initialized temporary variables.  Those arrays are kernel inputs in the
    finite element algorithm, not writable contraction intermediates, so
    combining them would overstate the working set created by factorization.
    """
    temporaries = tuple(
        kernel.ast.default_entrypoint.temporary_variables.values())
    mutable = [
        temporary for temporary in temporaries
        if temporary.shape
        and not (temporary.read_only and temporary.initializer is not None)
    ]
    tables = [
        temporary for temporary in temporaries
        if temporary.shape
        and temporary.read_only and temporary.initializer is not None
    ]
    mutable_sizes = [
        numpy.prod(temporary.shape, dtype=int) for temporary in mutable
    ]
    table_sizes = [
        numpy.prod(temporary.shape, dtype=int) for temporary in tables
    ]
    return types.SimpleNamespace(
        flops=kernel.flop_count,
        nscalar=sum(not temporary.shape for temporary in temporaries),
        nmutable=len(mutable_sizes),
        nmutable_elements=sum(mutable_sizes),
        largest_mutable=max(mutable_sizes, default=0),
        ntables=len(table_sizes),
        ntable_elements=sum(table_sizes),
        ast_lines=len(str(kernel.ast).splitlines()),
    )


def hottest_global_kernel(form: object, parameters: dict) -> tuple:
    """Assemble a form and return the global kernel that did most work.

    Parameters
    ----------
    form
        The form to assemble.
    parameters
        Form compiler parameters, which must match the ones the measured
        kernel was compiled with; ``assemble`` otherwise silently uses the
        defaults and every mode times the same generated code.

    Returns
    -------
    kernel
        The PyOP2 global kernel with the highest local flop count.
    comm
        Communicator it was called on.
    arguments
        Arguments it was called with.
    """
    from firedrake import assemble
    from pyop2.global_kernel import GlobalKernel

    calls = []
    original = GlobalKernel.__call__

    def record(self, comm, *arguments):
        calls.append((self, comm, arguments))
        return original(self, comm, *arguments)

    GlobalKernel.__call__ = record
    try:
        assemble(form, form_compiler_parameters=parameters)
    finally:
        GlobalKernel.__call__ = original

    return max(calls, key=lambda call: call[0].local_kernel.num_flops)


def time_kernel(form: object, parameters: dict, repeats: int) -> float:
    """Time repeated calls to the compiled cell kernel.

    Parameters
    ----------
    form
        The form to assemble.
    parameters
        Form compiler parameters.
    repeats
        Number of calls to average over.

    Returns
    -------
    float
        Mean seconds per call to the generated code.

    Notes
    -----
    Calling the compiled function directly measures the cell loop without
    the Python and PETSc work that surrounds a call to ``assemble``.
    """
    from pyop2.global_kernel import compile_global_kernel

    kernel, comm, arguments = hottest_global_kernel(form, parameters)
    execute = compile_global_kernel(kernel, comm)

    execute(*arguments)
    start = time.perf_counter()
    for _ in range(repeats):
        execute(*arguments)
    return (time.perf_counter() - start) / repeats


def dump_kernel(kernel: object, form: object, parameters: dict,
                directory: str, name: str) -> None:
    """Write the loopy and C forms of a kernel for inspection.

    Parameters
    ----------
    kernel
        Compiled TSFC kernel.
    form
        The form it came from.
    parameters
        Form compiler parameters.
    directory
        Directory to write into.
    name
        Basename identifying the case.

    Notes
    -----
    The local kernel shows the loop nest sum factorisation produced; the
    PyOP2 wrapper shows the C a compiler actually sees.
    """
    import loopy
    from pyop2.global_kernel import _generate_code_from_global_kernel

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{name}.loopy"), "w") as handle:
        handle.write(str(kernel.ast))
    with open(os.path.join(directory, f"{name}.c"), "w") as handle:
        handle.write(loopy.generate_code_v2(kernel.ast).device_code())

    global_kernel, comm, _ = hottest_global_kernel(form, parameters)
    with open(os.path.join(directory, f"{name}.wrapper.c"), "w") as handle:
        handle.write(_generate_code_from_global_kernel(global_kernel, comm))
