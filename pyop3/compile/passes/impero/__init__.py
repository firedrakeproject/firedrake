"""Impero to Impero passes that pyop3 applies before `pyop3.compile.mlir.MLIRBuilder` lowers the tree to MLIR."""

from __future__ import annotations

from gem import impero as imp
from gem.impero_utils import ImperoC

from pyop3.compile.passes.impero.item_loop import (
    assign_crossing_values, batch_items, flatten_item_temporaries, mark_atomics,
)
from pyop3.device import GPU, Device


def apply_passes(impero: ImperoC, device: Device, compiler_parameters) -> imp.Block:
    """Return the Impero tree of `impero` that `MLIRBuilder` lowers for `device`.

    Each item gets its own copy of the per-item temporaries first (`flatten_item_temporaries`).
    Then the passes depend on the device:

    * On the host, each value that one loop computes and another loop reads is stored
      in a temporary (`assign_crossing_values`).
    * On a GPU, the crossing values are stored as on the host, increments that other items
      can also do become atomic (`mark_atomics`), and each item runs on its own thread
      (`batch_items`). This is single-cell-per-thread (SCPT) parallelism.
    * On a GPU with the `thread_transposition` compiler parameter, the threads of a block
      share the work of each item (`thread_transpose`), with parameters for the target
      of `device`. This is an optimisation for high-order assembly.

    Parameters
    ----------
    impero
        The Impero code of the kernel, as `gem.impero_utils.compile_gem_new` returns it.
    device
        The device that the kernel runs on.
    compiler_parameters
        The parsed compiler parameters of the kernel.

    Returns
    -------
    imp.Block
        The tree for `MLIRBuilder`.

    Raises
    ------
    NotImplementedError
        If thread transposition is requested and does not support the kernel.
    """
    impero = flatten_item_temporaries(impero)
    if not isinstance(device, GPU):
        return assign_crossing_values(impero)
    if compiler_parameters.thread_transposition:
        from pyop3.compile.passes.impero.thread_transposition import thread_transpose

        return thread_transpose(impero.tree, target=device.target)
    return batch_items(mark_atomics(assign_crossing_values(impero)))
