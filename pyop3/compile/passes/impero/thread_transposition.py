"""Thread transposition as an Impero to Impero rewrite.

The item loop of a kernel (the only top-level loop with a dynamic extent) is
mapped onto GPU thread blocks, as in Knepley, Rupp and Terrel, "Finite element
integration with quadrature on the GPU" (2016). Each block processes a batch
of items. The work of an item is split into phases, and each phase is one loop
over the threads of the block with its own mapping of threads to points: for
example one thread per quadrature point, then one per basis function.

1. Normalisation (`normalise`). Each statement gets its own perfect loop
   nest, with summed loops innermost. Floating-point values that cross nests
   are stored in temporaries. Integer values are recomputed at each use.
   Temporaries that only copy other data are then propagated away, by
   `pyop3.compile.passes.impero.copy_propagation.propagate_copies`.
2. Phases (`fuse_phases`). Adjacent nests are fused when they run over
   the same domain and every dependence is pointwise.
3. Each phase becomes a `gem.impero.ThreadLoop` in a `gem.impero.ItemBatch` (`lower`).
4. Memory (`place_temporaries`). A temporary is kept in registers when
   one phase uses it and each thread accesses only its own entries. Otherwise
   it is in shared memory, with one copy per item of the batch. A shared
   temporary reuses the buffer of temporaries whose last phase is before its
   first phase. A shared buffer is a `temporary(..., shared=True)`.
5. An increment of a kernel argument is `"atomic"` unless it is injective
   (`pyop3.compile.passes.impero.item_loop.needs_atomic`).
6. Barriers are inserted on the MLIR, by `pyop3.compile.passes.mlir.thread_loops`.
7. The batch size and block size are chosen with a cost model (`choose_parameters`).

`thread_transpose` runs the stages in order.

The items must be independent, apart from increments of kernel arguments.
Each access to a per-item temporary must be a `gem.Indexed` with one
flat entry, such as `t[j]`, or `t[0]` for a scalar.
Item-independent work is not handled specially: the GEM scheduler already
hoists it out of the item loop.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from functools import cached_property

import numpy as np

from gem import gem, impero as imp, replace
from gem.node import pre_traversal

from pyop3 import utils
from pyop3.compile.passes.impero.copy_propagation import propagate_copies
from pyop3.compile.passes.impero.item_loop import needs_atomic, terminals, value_temporaries
from pyop3.compile.passes.impero.linear import LinearIndex, flat_entry, linear
from pyop3.compile.passes.impero.utils import (
    as_index, assign_value, gem_nodes, is_item_loop, is_temporary, name_generator, num_points, static_extent,
    temporary,
)
from pyop3.device import GPUTarget
from pyop3.dtypes import ScalarType


MAX_BATCH_SIZE = 128
"""The largest batch size that we try, haven't experimented beyond this """

COST_TOLERANCE = 0.02
"""Pairs within this relative cost of the cheapest are equal, and the smallest batch size wins.

The model cannot separate pairs with the same threads per item, such as
(32, 384) and (64, 768), but the larger batch measures a few percent slower.
"""

# Generalised cost weights from Nvidia GPUs from literature
# Other method would be fitting a model but this would obviously not be feasible
# (Microbenchmarking on Volta and Pascal Jia et al. 2018 and Abdelkhalik et al. 2022) 
FP64_LATENCY = 8  # cycles of an FP64 FMA on Volta to Ada
SHARED_LATENCY = 30  # cycles of a shared load on Volta to Ada 
GLOBAL_LATENCY = 200  # cycles of an L2 hit (same sources), as most global loads are cached maps, coordinates, tables
LSU_ACCESSES_PER_CYCLE = 16  # 64-bit accesses per SM per cycle: 32 banks of 4 bytes, 2 cycles per warp access
ISSUE_PER_CYCLE = 128  # thread instructions issued per SM per cycle: 4 schedulers, one warp instruction each


@dataclasses.dataclass(frozen=True)
class Parameters:
    batch_size: int
    block_size: int
    cost_per_item: float


# {{{ analysis data


@dataclasses.dataclass(frozen=True)
class Nest:
    """A perfect loop nest around one statement. `reductions` are the innermost loops of an increment."""

    loops: tuple[gem.Index, ...]
    reductions: tuple[gem.Index, ...]
    statement: imp.Assignment

    @property
    def atomic(self) -> bool:
        return self.statement.mode == "atomic"

    @property
    def domain(self) -> tuple[gem.Index, ...]:
        return tuple(i for i in self.loops if i not in self.reductions)

    @property
    def write(self) -> gem.Indexed:
        return self.statement.assignee

    @cached_property
    def reads(self) -> tuple[gem.Indexed, ...]:
        """The loads from variables. An increment also reads its assignee."""
        assignee = self.statement.assignee
        reads = tuple(
            node for node in pre_traversal((self.statement.expression, *assignee.indirect_children))
            if isinstance(node, gem.Indexed) and isinstance(node.aggregate, gem.Variable)
        )
        return reads + (assignee,) if self.statement.mode in ("inc", "atomic") else reads

    @property
    def accesses(self) -> tuple[gem.Indexed, ...]:
        """The write first, then the reads."""
        return (self.write, *self.reads)

    def accesses_to(self, name: str) -> tuple[gem.Indexed, ...]:
        return tuple(a for a in self.accesses if a.aggregate.name == name)

    def rewritten(self, mapping: Mapping) -> Nest:
        s = self.statement
        statement = imp.Assignment(replace(s.assignee, mapping), replace(s.expression, mapping), s.mode)
        return dataclasses.replace(self, statement=statement)


@dataclasses.dataclass(frozen=True)
class Phase:
    """Adjacent nests over one domain with pointwise dependences, renamed onto `domain`."""

    domain: tuple[gem.Index, ...]
    nests: tuple[Nest, ...]

    @cached_property
    def accesses(self) -> tuple[gem.Indexed, ...]:
        return tuple(a for nest in self.nests for a in nest.accesses)

    def accesses_to(self, name: str) -> tuple[gem.Indexed, ...]:
        return tuple(a for a in self.accesses if a.aggregate.name == name)

    @property
    def width(self) -> int:
        return num_points(self.domain)


@dataclasses.dataclass(frozen=True)
class Temporary:
    """A temporary in the registers of its only `phase`, or in shared memory if `phase` is `None`.

    `size` is the number of entries per thread in registers, or per item in shared memory.
    The `variable` and `size` of a shared temporary are those of its buffer, which other temporaries can share.
    """

    variable: gem.Variable
    phase: int | None
    size: int

    @property
    def in_registers(self) -> bool:
        return self.phase is not None

    def placed(self, batch_size: int) -> gem.Variable:
        size = self.size if self.in_registers else batch_size * self.size
        return temporary(self.variable.name, (size,), self.variable.dtype or ScalarType, shared=not self.in_registers)


# }}}


def thread_transpose(
    tree: imp.Block,
    *,
    target: GPUTarget,
    batch_size: int | None = None,
    block_size: int | None = None,
    extra_shared_bytes: int = 0,
) -> imp.Block:
    """Return `tree` with its item loop replaced by an `ItemBatch`.

    `batch_size` (N_c) and `block_size` (T) fix the parameters instead of
    the choice of the cost model. `target` gives the limits and the FP64 rate of the device.
    `extra_shared_bytes` is the shared memory per batch that is allocated outside this pass.

    Raises
    ------
    NotImplementedError
        If thread transposition does not support the kernel.
    """
    try:
        loop = _item_loop(tree)
        phases = fuse_phases(propagate_copies(normalise(tree, loop)))
        _check(tree, loop, phases)
        temporaries = place_temporaries(phases, loop.index)
        counts = tuple(_phase_counts(phase, temporaries) for phase in phases)
        shared = sum(map(_bytes, {t.variable for t in temporaries.values() if not t.in_registers}))
        parameters = choose_parameters(
            phases, counts, shared, extra_shared_bytes, target, batch_size=batch_size, block_size=block_size
        )
        return lower(tree, loop, phases, temporaries, parameters)
    except NotImplementedError as e:
        raise NotImplementedError(
            f"{e}. This kernel is not supported by thread transposition; "
            "run it without the thread_transposition compiler parameter."
        ) from e


def _item_loop(tree: imp.Block) -> imp.For:
    loops = [n for n in tree.children if is_item_loop(n)]
    if len(loops) != 1:
        raise ValueError(f"Expected exactly one top-level item loop, found {len(loops)}")
    return loops[0]


# {{{ stage 1: normalisation

def normalise(tree: imp.Block, loop: imp.For) -> tuple[Nest, ...]:
    """The normalised nests of the body of `loop`, in execution order."""
    statements = [(tuple(f.index for f in loops), terminal) for loops, terminal in terminals(loop.statement)]
    temporaries = value_temporaries(
        statements, name_generator(tree), "tt",
        lambda loops, value: None if _is_inlined(value) else tuple(i for i in loops if i in value.free_indices),
    )
    return tuple(
        _nest(loops, statement, loop.index) for loops, terminal in statements
        if (statement := _as_assignment(terminal, temporaries)) is not None
    )


def _as_assignment(terminal: imp.Terminal, temporaries) -> imp.Assignment | None:
    """Return `terminal` as an assignment that reads crossing values from `temporaries`, or `None` to drop it."""
    if (assignment := assign_value(terminal, temporaries)) is not None:
        return assignment
    match terminal:
        case imp.Evaluate() | imp.Noop():
            return None
        case imp.Assignment(mode="write" | "inc"):
            return imp.Assignment(
                replace(terminal.assignee, temporaries), replace(terminal.expression, temporaries), terminal.mode
            )
    raise NotImplementedError(f"{terminal!r} is not supported")


def _nest(loops: tuple[gem.Index, ...], statement: imp.Assignment, item: gem.Index) -> Nest:
    """Put `statement` in a nest with its summed loops innermost, and mark it atomic if needed.

    The summation order is unchanged.
    """
    free = statement.assignee.free_indices
    reductions = tuple(i for i in loops if i not in free) if statement.mode == "inc" else ()
    domain = tuple(i for i in loops if i not in reductions)
    if needs_atomic(statement, (item, *domain)):
        statement = imp.Assignment(statement.assignee, statement.expression, "atomic")
    return Nest(domain + reductions, reductions, statement)

# }}}


# {{{ stage 2: phases

def fuse_phases(nests: tuple[Nest, ...]) -> tuple[Phase, ...]:
    """Greedily fuse each nest into the previous phase, or open a new one."""
    phases = []
    for nest in nests:
        if phases and (fused := _try_fuse(phases[-1], nest)) is not None:
            phases[-1] = fused
        else:
            phases.append(Phase(nest.domain, (nest,)))
    return tuple(phases)


def _check(tree: imp.Block, loop: imp.For, phases: tuple[Phase, ...]) -> None:
    temporaries = {e.name for e in gem_nodes(loop.statement) if is_temporary(e)}
    outside = {
        e.name for e in gem_nodes(*(n for n in tree.children if n is not loop))
        if isinstance(e, gem.Variable)
    }
    if names := outside & temporaries:
        raise NotImplementedError(f"Temporaries accessed inside and outside the item loop: {sorted(names)}")
    # An indirect write, such as t[map[i]], is assumed injective, as the maps of one item are
    for phase in phases:
        for w in (nest.write for nest in phase.nests):
            if is_temporary(w.aggregate) and (index := linear(w)) and not index.is_injective(phase.domain):
                raise NotImplementedError(f"{w.aggregate.name}[{flat_entry(w.multiindex)}] is written by more than one thread of {phase.domain}")

# }}}


# {{{ stages 3 and 4: memory and lowering

def place_temporaries(phases: tuple[Phase, ...], item: gem.Index) -> dict[str, Temporary]:
    """Keep a temporary in registers if one phase uses it and each thread accesses only its own entries.

    Each other temporary takes over the shared buffer of the same dtype that fits it best, among the
    buffers whose temporaries are no longer used, or else a new buffer.
    """
    variables = {a.aggregate.name: a.aggregate for phase in phases for a in phase.accesses if is_temporary(a.aggregate)}
    temporaries, lifetimes = {}, {}
    for name, variable in variables.items():
        uses = [k for k, phase in enumerate(phases) if phase.accesses_to(name)]
        domain, accesses = phases[uses[0]].domain, phases[uses[0]].accesses_to(name)
        if len(uses) == 1 and _has_private_layout(accesses, domain, item):
            temporaries[name] = Temporary(variable, uses[0], max(linear(a).without(domain).size for a in accesses))
        else:
            lifetimes[name] = (uses[0], uses[-1])

    buffers = []  # [names, size, last phase, dtype]
    for name, (first, last) in sorted(lifetimes.items(), key=lambda item: item[1]):
        (size,), dtype = variables[name].shape, variables[name].dtype
        free = [b for b in buffers if b[2] < first and b[3] == dtype]
        fitting = [b for b in free if b[1] >= size]
        if fitting or free:
            buffer = min(fitting, key=lambda b: b[1]) if fitting else max(free, key=lambda b: b[1])
        else:
            buffer = [[], 0, last, dtype]
            buffers.append(buffer)
        buffer[0].append(name)
        buffer[1:3] = max(buffer[1], size), last
    for names, size, _, dtype in buffers:
        variable = temporary(names[0], (size,), dtype)
        temporaries |= dict.fromkeys(names, Temporary(variable, None, size))
    return temporaries


def _has_private_layout(accesses: tuple[gem.Indexed, ...], domain: tuple[gem.Index, ...], item: gem.Index) -> bool:
    """Whether every access is linear without the item index, with the same nonzero coefficient per domain index."""
    parts = []
    for a in accesses:
        if (index := linear(a)) is None or item in index.coefficients:
            return False
        parts.append({i: index.coefficients.get(i, 0) for i in domain})
    return all(all(p.values()) and p == parts[0] for p in parts)


def lower(
    tree: imp.Block, loop: imp.For, phases: tuple[Phase, ...], temporaries: Mapping[str, Temporary], parameters: Parameters,
) -> imp.Block:
    """Replace `loop` by an `ItemBatch` with one `ThreadLoop` per phase.

    A register keeps the entries of one point of its phase. A shared
    temporary has one copy per item, the copy of item `c` at `c * size`.
    """
    local = gem.Index(name="local", extent=parameters.batch_size)
    placed = {name: t.placed(parameters.batch_size) for name, t in temporaries.items()}

    def relocate(access: gem.Indexed) -> gem.Indexed:
        t = temporaries[access.aggregate.name]
        if t.in_registers:
            index = linear(access).without(phases[t.phase].domain).to_gem()
        else:
            offset = local * gem.as_gem_uint(t.size) if t.size != 1 else local
            index = offset + gem.as_gem_uint(flat_entry(access.multiindex))
        return gem.Indexed(placed[access.aggregate.name], (as_index(index),))

    # shortcut: a relocated access keeps any temporary in its own index unrelocated, rewrite bottom-up if one appears
    relocated = {a: relocate(a) for phase in phases for a in phase.accesses if a.aggregate.name in temporaries}
    body = []
    for k, phase in enumerate(phases):
        statements = imp.Block(
            _loops_around(
                imp.Assignment(
                    replace(nest.statement.assignee, relocated),
                    replace(nest.statement.expression, relocated),
                    nest.statement.mode,
                ),
                nest.reductions,
            )
            for nest in phase.nests
        )
        body.append(imp.ThreadLoop(phase.domain, statements))

    batch = imp.ItemBatch(loop.index, local, batch_size=parameters.batch_size, block_size=parameters.block_size, statement=imp.Block(body))
    # The other top-level nodes only compute item-independent values, which the thread loops recompute
    return imp.Block(batch if node is loop else node for node in tree.children)

# }}}


# {{{ stage 7: parameters

def choose_parameters(
    phases: tuple[Phase, ...],
    counts: tuple[np.ndarray, ...],
    shared_per_item: int,
    shared_per_batch: int,
    target: GPUTarget,
    *,
    batch_size: int | None = None,
    block_size: int | None = None,
) -> Parameters:
    """The cheapest pair, among every N_c up to :data:`MAX_BATCH_SIZE` and every T a multiple of the warp size."""
    batch_sizes = range(1, MAX_BATCH_SIZE + 1) if batch_size is None else (batch_size,)
    block_sizes = (
        range(target.warp_size, target.max_threads_per_block + 1, target.warp_size)
        if block_size is None else (block_size,)
    )
    w, fp64 = target.warp_size, target.fp64_per_cycle
    op_latency = max(FP64_LATENCY, w * 4 / fp64)  # a warp's FP64 operation on one quarter of the SM

    def cost_per_item(n: int, t: int) -> float | None:
        """The estimated SM cycles of one item, or `None` if the pair does not fit the target.

            Σ_k max(B ⌈N_c n_k / W⌉ W r_k,  ⌈N_c n_k / T⌉ l_k) / (B N_c)

        with n_k the width of phase k, W the warp size and B the resident blocks
        per SM. r_k is the cycles of a point on the busiest unit of the SM:
        FP64 (operations), load/store (global and shared accesses) or issue
        (all of them). l_k is the latency of a point. A barrier ends each phase, so a
        phase lasts as long as the work of the SM or the critical path of a block,
        whichever is longer. Every SM is assumed to hold B blocks.
        """
        shared = n * shared_per_item + shared_per_batch
        if shared > target.shared_memory_per_block or t > target.max_threads_per_block:
            return None
        blocks = min(target.max_blocks_per_sm, target.max_threads_per_sm // t)
        if shared:
            blocks = min(blocks, target.shared_memory_per_sm // shared)
        if blocks == 0:
            return None

        cycles = 0.0
        for phase, (ops, glob, sh) in zip(phases, counts, strict=True):
            points = n * phase.width
            rate = max(ops / fp64, (glob + sh) / LSU_ACCESSES_PER_CYCLE, (ops + glob + sh) / ISSUE_PER_CYCLE)
            work = blocks * utils.ceildiv(points, w) * w * rate
            path = utils.ceildiv(points, t) * (ops * op_latency + sh * SHARED_LATENCY + glob * GLOBAL_LATENCY)
            cycles += max(work, path)
        return cycles / (blocks * n)

    pairs = {(n, t): cost for n in batch_sizes for t in block_sizes if (cost := cost_per_item(n, t)) is not None}
    if not pairs:
        raise ValueError(f"No batch size and block size fit the target with {shared_per_item} bytes of shared memory per item")
    lowest = min(pairs.values())
    pair = min(pair for pair, cost in pairs.items() if cost <= (1 + COST_TOLERANCE) * lowest)
    return Parameters(*pair, pairs[pair])


def _phase_counts(phase: Phase, temporaries: Mapping[str, Temporary]) -> np.ndarray:
    """The operations, global accesses and shared accesses of one point of `phase`, times the reductions.

    An atomic is a global access, as only kernel arguments are atomic. Register accesses are not counted.
    """
    def access(variable: gem.Variable) -> tuple[int, int, int]:
        t = temporaries.get(variable.name)
        return (0, 1, 0) if t is None else (0, 0, 0) if t.in_registers else (0, 0, 1)

    total = np.zeros(3)
    for nest in phase.nests:
        s = nest.statement
        counts = np.array(access(nest.write.aggregate))
        for node in pre_traversal((s.expression, *s.assignee.indirect_children)):
            if isinstance(node, (gem.Operator, gem.MathFunction, gem.Comparison, gem.Conditional)):
                counts[0] += 1
            elif isinstance(node, gem.Indexed) and isinstance(node.aggregate, gem.Variable):
                counts += access(node.aggregate)
        total += counts * num_points(nest.reductions)
    return total

# }}}


# {{{ phase fusion

def _try_fuse(phase: Phase, nest: Nest) -> Phase | None:
    """Return `phase` with `nest` fused in, or `None` if this is not legal.

    The nest domain must map onto the phase domain by an extent-preserving
    bijection σ. Every variable that couples them (RAW, WAR or WAW) must be
    accessed linearly, injectively in the phase, and identically under σ, so
    every dependence is between the same point. Accesses with disjoint ranges
    and commuting atomic increments do not couple.
    """
    if sorted(map(static_extent, nest.domain)) != sorted(map(static_extent, phase.domain)):
        return None
    written = {n.write.aggregate.name for n in phase.nests}
    read = {a.aggregate.name for n in phase.nests for a in n.reads}
    names = ({a.aggregate.name for a in nest.accesses} & written) | ({nest.write.aggregate.name} & read)
    coupled = {
        name for name in names
        if not _disjoint(phase, nest, name) and not _atomic_increments_only(phase.nests + (nest,), name)
    }
    sigma = {}
    for name in coupled:
        theirs = [linear(a) for a in phase.accesses_to(name)]
        ours = [linear(a) for a in nest.accesses_to(name)]
        if None in ours or any(g is None or not g.is_injective(phase.domain) for g in theirs):
            return None
        if not all(_bind(f, g, sigma, nest.domain, phase.domain) for f in ours for g in theirs):
            return None

    # Map the unbound domain indices, preferring the identity
    unbound = [i for i in phase.domain if i not in sigma.values()]
    for index in nest.domain:
        if index not in sigma:
            candidates = [i for i in unbound if static_extent(i) == static_extent(index)]
            sigma[index] = index if index in candidates else candidates[0]
            unbound.remove(sigma[index])
    return Phase(phase.domain, phase.nests + (_rename(nest, sigma, phase.domain),))


def _disjoint(phase: Phase, nest: Nest, name: str) -> bool:
    ours = [linear(a) for a in nest.accesses_to(name)]
    theirs = [linear(a) for a in phase.accesses_to(name)]
    if any(f is None or f.bounds is None for f in ours + theirs):
        return False
    return all(f.bounds[1] < g.bounds[0] or g.bounds[1] < f.bounds[0] for f in ours for g in theirs)


def _bind(ours: LinearIndex, theirs: LinearIndex, sigma: dict, our_domain: tuple, their_domain: tuple) -> bool:
    """Whether `ours` equals `theirs` after renaming by `sigma`, which is extended in place.

    `theirs` is injective over `their_domain`, so a coefficient determines
    the binding of a domain index. Other indices only match themselves.
    """
    if ours.constant != theirs.constant:
        return False
    by_coefficient = {k: i for i, k in theirs.coefficients.items() if i in their_domain}
    for index, k in ours.coefficients.items():
        if index in our_domain:
            if index not in sigma:
                target = by_coefficient.get(k)
                if target is None or target in sigma.values() or static_extent(target) != static_extent(index):
                    return False
                sigma[index] = target
        elif index in their_domain:
            return False
    return {sigma.get(i, i): k for i, k in ours.coefficients.items()} == theirs.coefficients


def _atomic_increments_only(nests: tuple[Nest, ...], name: str) -> bool:
    """Whether every nest that accesses kernel argument `name` only increments it indirectly, so atomically."""
    for nest in nests:
        if accesses := nest.accesses_to(name):
            w = nest.write
            if not nest.atomic or w.aggregate.name != name or linear(w) is not None or len(accesses) != 2:
                return False
    return True


def _rename(nest: Nest, sigma: Mapping[gem.Index, gem.Index], domain: tuple[gem.Index, ...]) -> Nest:
    """Rename the domain of `nest` by `sigma`, and any reduction index that clashes with `domain`."""
    mapping = dict(sigma) | {i: gem.Index(extent=i.extent) for i in nest.reductions if i in domain}
    nest = nest.rewritten(mapping)
    rename = lambda indices: tuple(mapping.get(i, i) for i in indices)  # noqa: E731
    return dataclasses.replace(nest, loops=rename(nest.loops), reductions=rename(nest.reductions))

# }}}


# {{{ helpers

def _is_inlined(expr: gem.Node) -> bool:
    """Whether `expr` is recomputed at each use instead of stored.

    Tables are read from their globals; indices, constants and loads are
    cheap; integer values, such as map offsets, keep their map loads in the
    phase that uses them. An index sum is always stored.
    """
    if expr.shape or isinstance(expr, (gem.Index, gem.Constant, gem.Variable)):
        return True
    if isinstance(expr, gem.Indexed) and isinstance(expr.aggregate, (gem.Variable, gem.Constant)):
        return True
    return not isinstance(expr, gem.IndexSum) and np.dtype(expr.dtype or ScalarType).kind in "iub"


def _loops_around(statement: imp.Node, indices: tuple[gem.Index, ...]) -> imp.Node:
    for index in reversed(indices):
        statement = imp.For(index, imp.Block([statement]))
    return statement


def _bytes(variable: gem.Variable) -> int:
    return int(np.prod(variable.shape)) * np.dtype(variable.dtype or ScalarType).itemsize

# }}}
