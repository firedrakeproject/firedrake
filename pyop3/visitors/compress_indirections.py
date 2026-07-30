from __future__ import annotations

import collections
import contextlib
import functools
import itertools
import numbers

from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3.axis_tree
import pyop3.collections
import pyop3.expr
import pyop3.expr.visitors
import pyop3.index_tree
import pyop3.insn
import pyop3.node
from pyop3 import utils

# TODO: account for non-affine accesses in arrays and selectively apply this
INDIRECTION_PENALTY_FACTOR = 5

MINIMUM_COST_TABULATION_THRESHOLD = 128
"""The minimum cost below which tabulation will not be considered.

Indirections with a cost below this are considered as fitting into cache and
so memory optimisations are ineffectual.

"""

MAX_COST_CONSIDERATION_FACTOR = 3
"""Maximum factor an expression cost can exceed the minimum and still be considered."""


# TODO: This isn't really a visitor like this...
@PETSc.Log.EventDecorator()
def materialize_indirections(insn: pyop3.insn.Instruction, *, compress: bool = False) -> pyop3.insn.Instruction:
    if compress:
        selector = _select_candidate_indirections_compress
    else:
        selector = _select_candidate_indirections_nocompress
    materialize_idxs = selector(insn)

    # Eagerly return if there are no symbolic dats to materialise and insert
    if not materialize_idxs:
        return insn

    # Now materialize the right bits and return
    return _insert_materialized_indirections(insn, materialize_idxs)


def _select_candidate_indirections_compress(insn):
    # This optimisation is collective but since the array size is part of the
    # heuristic one can get differing optimisation choices on different ranks. We
    # therefore perform all the heuristics on rank 0 and broadcast the selections.
    return pyop3.mpi.safe_noncollective(
        insn.comm,
        lambda: _select_candidate_indirections_compress_serial(insn),
        root=0,
    )

def _select_candidate_indirections_compress_serial(insn):
    candidates = _collect_candidate_indirections(insn, compress=True)
    candidates = _trim_candidates(candidates)

    # Optimise the search tree by only considering disjoint subsets of
    # candidates. For example, if we have candidates
    #
    #     {a: [A, B, C, D], b: [X, Y]}
    #
    # we can speed things up by only investigating 4+2 options instead
    # of 4*2.
    disjoint_subsets: list[tuple[dict, set]] = []
    for terminal_id, terminal_candidates in candidates.items():
        # terminal_symdats = {
        #     symdat
        #     for _, _, symdats in terminal_candidates
        #     for _, symdat, _ in symdats
        # }
        terminal_symdats = {
            loc
            for _, _, symdats in terminal_candidates
            for loc, _, _ in symdats
        }
        disjoint_subsets.append(
            ({terminal_id: terminal_candidates}, terminal_symdats)
        )
    # Fixed point iteration to ensure subsets are fully disjoint
    while True:
        new_disjoint_subsets = []
        for terminal_id, terminal_candidates in candidates.items():
            # terminal_symdats = {
            #     symdat
            #     for _, _, symdats in terminal_candidates
            #     for _, symdat, _ in symdats
            # }
            terminal_symdats = {
                loc
                for _, _, symdats in terminal_candidates
                for loc, _, _ in symdats
            }
            for subset_candidates, subset_symdats in new_disjoint_subsets:
                if terminal_symdats.intersection(subset_symdats):
                    subset_candidates[terminal_id] = terminal_candidates
                    subset_symdats.update(terminal_symdats)
                    break
            else:
                # not found in an existing subset, create a new one
                new_disjoint_subsets.append(
                    ({terminal_id: terminal_candidates}, terminal_symdats)
                )

        if new_disjoint_subsets == disjoint_subsets:
            break
        else:
            disjoint_subsets = new_disjoint_subsets

    # Now select the combination with the lowest combined cost. We can make savings here
    # by sharing indirection maps between different arguments. For example, if we have
    #
    #     dat1[mapA[mapB[mapC[i]]]]
    #     dat2[mapB[mapC[i]]]
    #
    # then we can (sometimes) minimise the data cost by having
    #     dat1[mapA[mapBC[i]]]
    #     dat2[mapBC[i]]
    #
    # instead of
    #
    #     dat1[mapABC[i]]
    #     dat2[mapBC[i]]
    materialize_idxs = pyop3.collections.OrderedSet()
    for candidate_subset, _ in disjoint_subsets:
        best_subset_symdat_locs = None
        min_subset_cost = None
        for shared_candidate in utils.expand_collection_of_iterables(candidate_subset):
            symdat_costs = {
                loc: cost
                for _, _, symdats in shared_candidate.values()
                for loc, _, cost in symdats
            }

            cost = 0
            for _, cost_expr, symdats in shared_candidate.values():
                cost += pyop3.expr.visitors.evaluate(cost_expr, name_vars=symdat_costs)
                # Subsequent expressions can reuse the array, so cost nothing
                for loc, _, _ in symdats:
                    symdat_costs[loc] = 0

            if best_subset_symdat_locs is None or cost < min_subset_cost:
                best_subset_symdat_locs = pyop3.collections.OrderedSet([
                    loc
                    for _, _, symdats in shared_candidate.values()
                    for loc, _, _ in symdats
                ])
                min_subset_cost = cost
        assert best_subset_symdat_locs is not None
        materialize_idxs.update(best_subset_symdat_locs)

    return pyop3.collections.OrderedFrozenSet(materialize_idxs)


def _trim_candidates(candidates):
    # Find the best cost if we look at each terminal independently. This
    # provides a useful upper bound.
    max_cost = 0
    for terminal_id, terminal_candidates in candidates.items():
        min_terminal_cost = min(
            (cost for cost, _, _ in terminal_candidates), default=0
        )
        max_cost += min_terminal_cost
    assert isinstance(max_cost, numbers.Integral)

    # Drop any immediately bad candidates (that exceed this cost). Also
    # skip over any terminals that don't generate any symbolic dats.
    trimmed_candidates = {}
    for terminal_id, terminal_candidates in candidates.items():
        if len(terminal_candidates) == 0:
            continue

        trimmed_terminal_candidates = []
        min_terminal_cost = min((cost for cost, _, _ in terminal_candidates), default=0)
        for cost, cost_expr, sym_dats in terminal_candidates:
            if cost <= max_cost and cost <= min_terminal_cost * MAX_COST_CONSIDERATION_FACTOR:
                trimmed_terminal_candidates.append((cost, cost_expr, sym_dats))
        trimmed_candidates[terminal_id] = trimmed_terminal_candidates
    return trimmed_candidates


def _select_candidate_indirections_nocompress(insn):
    candidates = _collect_candidate_indirections(insn, compress=False)

    symdat_locs = pyop3.collections.OrderedSet()
    for terminal_id, terminal_candidates in candidates.items():
        if not terminal_candidates:
            continue

        _, _, terminal_symdats_info = utils.just_one(terminal_candidates)
        for loc, _, _ in terminal_symdats_info:
            symdat_locs.add(loc)
    return pyop3.collections.OrderedFrozenSet(symdat_locs)


class _CandidateIndirectionsCollector(pyop3.node.NodeVisitor):

    def __init__(self):
        self._collecting = False
        super().__init__()

    @contextlib.contextmanager
    def collecting(self):
        """Context manager used when we are actually collecting candidate indirections.

        When this context is active we want flat iterables back, not mappings.

        """
        prev = self._collecting
        self._collecting = True
        yield
        self._collecting = prev

    def get_cache_key(self, node, **kwargs):
        return (*super().get_cache_key(node, **kwargs), self._collecting)

    # TODO dont need this any more, just access self.index
    def preprocess_node(self, node) -> tuple[Any, ...]:
        return node, self.index

    @functools.singledispatchmethod
    def process(self, obj: pyop3.obj.Object, /, *args, **kwargs) -> tuple[tuple[Any, int, int], ...]:
        utils.raise_missing_dispatch_handler(obj)

    def _null(self, obj: Any, index, /, **kwargs):
        """Handler for terminals where we don't do anything."""
        if self._collecting:
            return ()
        else:
            return idict({})

    # {{{ pyop3.expr

    @process.register
    def _(self, op: pyop3.expr.BinaryOperator, index, /, *, compress: bool) -> tuple:
        if not self._collecting:
            return utils.merge_dicts(
                self(x, compress=compress) for x in op.operands
            )

        operand_candidatess = tuple(
            self(o, compress=compress) for o in op.operands
        )
        candidates = []
        for operand_candidates in itertools.product(*operand_candidatess):
            cost_per_operand, cost_expr_per_operand, symdats_per_operand = \
                zip(*operand_candidates, strict=True)

            # If there is at most one non-zero operand cost then there is no point
            # in compressing the expression.
            if sum(cost > 0 for cost in cost_per_operand) <= 1:
                compress = False

            # NOTE: This isn't quite correct. For example consider the expression
            # 'mapA[i] + mapA[i]'. The cost is just the cost of 'mapA[i]', not double.
            candidate_cost = sum(cost_per_operand)
            candidate_cost_expr = sum(cost_expr_per_operand)
            candidate_symdats = sum(symdats_per_operand, ())
            candidates.append((candidate_cost, candidate_cost_expr, candidate_symdats))

        if compress:
            # Now also include a candidate representing the packing of the expression
            # into a Dat. The cost for this is simply the size of the resulting array.
            # Only do this when the cost is large as small arrays will fit in cache
            # and not benefit from the optimisation.
            if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for cost, _, _ in candidates):
                op_axes = utils.just_one(pyop3.expr.visitors.get_shape(op))
                op_loop_axes = pyop3.expr.visitors.get_loop_axes(op)

                op_cost = op_axes.local_size
                for loop_axes in op_loop_axes.values():
                    for loop_axis in loop_axes:
                        op_cost *= loop_axis.component.local_size
                if not isinstance(op_cost, numbers.Integral):
                    raise NotImplementedError("Ragged sizes are not supported")


                cost_expr = pyop3.expr.NameVar(self.index)
                symdat = pyop3.expr.CompositeDat(op_axes, {op_axes.leaf_path: op})
                symdat_info = (self.index, symdat, op_cost)
                candidates.append((op_cost, cost_expr, (symdat_info,)))

        return tuple(candidates)

    @process.register
    def _(
        self,
        expr: pyop3.expr.LinearDatBufferExpression,
        index,
        /,
        *,
        compress: bool,
    ):
        if not self._collecting:
            with self.collecting():
                return idict({self.index: self(expr.layout, compress=compress)})

        # The cost of an expression dat (i.e. the memory volume) is given by... TODO
        # Remember that the axes here described the outer loops that exist and that
        # index expressions that do not access data (e.g. 2i+j) have a cost of zero.
        # dat[2i+j] would have a cost equal to ni*nj as those would be the outer loops
        # TODO: can this be nicer? I have other routines for getting shapes from expressions
        dat_axes = utils.just_one(pyop3.expr.visitors.get_shape(expr.layout))
        dat_loop_axes = pyop3.expr.visitors.get_loop_axes(expr.layout)
        dat_cost = dat_axes.local_size
        for loop_axes in dat_loop_axes.values():
            for loop_axis in loop_axes:
                dat_cost *= loop_axis.component.local_size
        if not isinstance(dat_cost, numbers.Integral):
            raise NotImplementedError("Ragged sizes are not supported")

        candidates = []
        layout_candidates = self(expr.layout, compress=compress)
        for layout_cost, layout_cost_expr, layout_symdats in layout_candidates:
            # TODO: Only apply penalty for non-affine layouts that actually involve an indirection
            candidate_cost = dat_cost + layout_cost * INDIRECTION_PENALTY_FACTOR
            candidate_cost_expr = dat_cost + layout_cost_expr * INDIRECTION_PENALTY_FACTOR
            candidates.append((candidate_cost, candidate_cost_expr, layout_symdats))

        if compress:
            if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for cost, _, _ in candidates):
                # We use a symbolic expression for the overall cost because we
                # need to be able to replace costs with zeros if we end up sharing
                # materialised dats between terminals.
                cost_expr = pyop3.expr.NameVar(self.index)
                symdat = pyop3.expr.CompositeDat(dat_axes, {dat_axes.leaf_path: expr})
                symdat_info = (self.index, symdat, dat_cost)
                candidates.append((dat_cost, cost_expr, (symdat_info,)))

        return tuple(candidates)

    @process.register
    def _(
        self,
        dat_expr: pyop3.expr.NonlinearDatBufferExpression,
        index,
        /,
        **kwargs,
    ) -> idict:
        assert not self._collecting

        return utils.merge_dicts(
            self(l, **kwargs)
            for l in dat_expr.layouts.values()
        )

    @process.register
    def _(self, tern: pyop3.expr.TernaryOperator, index, /, **kwargs) -> idict:
        return utils.merge_dicts(self(x, **kwargs) for x in tern.operands)

    @process.register(pyop3.expr.AxisVar)
    @process.register(pyop3.expr.LoopIndexVar)
    @process.register(pyop3.expr.Scalar)
    @process.register(pyop3.expr.ScalarBufferExpression)
    @process.register(pyop3.expr.OpaqueTerminal)
    @process.register(pyop3.expr.NaN)
    def _(self, var, *args, **kwargs):
        return self._null(var, *args, **kwargs)

    @process.register
    def _(self, mat_expr: pyop3.expr.MatPetscMatBufferExpression, index, /, *, compress: bool) -> idict:
        candidates = {}
        layouts = [mat_expr.row_layout, mat_expr.column_layout]
        for i, layout in enumerate(layouts):
            assert isinstance(layout, pyop3.expr.CompositeDat)
            op_axes = utils.just_one(pyop3.expr.visitors.get_shape(layout))
            op_cost = pyop3.expr.visitors.loopified_shape(layout)[0].local_size
            # op_loop_axes = pyop3.expr.visitors.get_loop_axes(layout)
            #
            # op_cost = op_axes.local_size
            # for loop_axes in op_loop_axes.values():
            #     for loop_axis in loop_axes:
            #         op_cost *= loop_axis.component.local_size
            # if not isinstance(op_cost, numbers.Integral):
            #     raise NotImplementedError("Ragged sizes are not supported")

            cost_expr = pyop3.expr.NameVar((self.index, i))
            # symdat = pyop3.expr.CompositeDat(op_axes, {op_axes.leaf_path: op})
            symdat = layout
            symdat_info = ((self.index, i), symdat, op_cost)
            candidates[self.index, i] = ((op_cost, cost_expr, (symdat_info,)),)
        return idict(candidates)

    @process.register
    def _(self, mat_expr: pyop3.expr.MatArrayBufferExpression, index, /, *,  compress: bool) -> idict:
        candidates = {}
        with self.collecting():
            layoutss = [mat_expr.row_layouts, mat_expr.column_layouts]
            for i, layouts in enumerate(layoutss):
                for j, (path, layout) in enumerate(layouts.items()):
                    candidates[self.index, i, j] = self(
                        layout, compress=compress
                    )
        return idict(candidates)

    # }}}

    # {{{ pyop3.insn

    @process.register(pyop3.insn.NullInstruction)
    @process.register(pyop3.insn.Exscan)  # assume we are fine
    def _(self, null: pyop3.insn.InstructionList, index, /, **kwargs) -> idict:
        return idict()


    @process.register
    def _(self, insn_list: pyop3.insn.InstructionList, index, /, **kwargs) -> idict:
        return utils.merge_dicts(
            (self(insn, **kwargs) for insn in insn_list),
        )

    @process.register(pyop3.insn.Loop)
    def _(self, loop: pyop3.insn.Loop, index, /, **kwargs) -> idict:
        return utils.merge_dicts(
            (
                self(stmt, **kwargs)
                for stmt in loop.statements
            ),
        )

    @process.register
    def _(self, terminal: pyop3.insn.NonEmptyTerminal, index, /, *, compress: bool) -> idict:
        candidates = {}
        for i, arg in enumerate(terminal.arguments):
            per_arg_candidates = self(arg, compress=compress)
            candidates |= per_arg_candidates
        return idict(candidates)

    # }}}

    # {{{ misc

    @process.register
    def _(self, var: numbers.Number, *args, **kwargs):
        return self._null(var, *args, **kwargs)

    # }}}


def _collect_candidate_indirections(
    obj: pyop3.obj.Object,
    *,
    compress: bool,
):
    collector = _get_candidate_indirections_collector(obj.comm)
    collector.index = ()  # reset counter
    collector._index_stack = collections.defaultdict(itertools.count)
    return collector(obj, compress=compress)


@pyop3.cache.memory_cache(heavy=True)
def _get_candidate_indirections_collector(comm):
    return _CandidateIndirectionsCollector()


class _MaterializedIndirectionsInserter(pyop3.node.NodeVisitor):

    def __init__(self, comm):
        self.comm=comm
        self._linear = False
        super().__init__()

    @contextlib.contextmanager
    def enforce_linear(self):
        prev = self._linear
        self._linear = True
        yield
        self._linear = prev

    def get_cache_key(self, node, **kwargs):
        return (*super().get_cache_key(node, **kwargs), self._linear)

    @functools.singledispatchmethod
    def process(self, obj: pyop3.obj.Object, /, *args, **kwargs) -> Never:
        utils.raise_missing_dispatch_handler(obj)

    # {{{ pyop3.expr

    @process.register
    def _(self, op: pyop3.expr.BinaryOperator, /, *, materialize_idxs) -> tuple:
        if self.index in materialize_idxs:
            assert self._linear

            op_axes = utils.just_one(pyop3.expr.visitors.get_shape(op))
            symdat = pyop3.expr.CompositeDat(op_axes, {op_axes.leaf_path: op})
            return pyop3.expr.visitors.materialize_composite_dat(symdat, self.comm, linear=True)

        else:
            new_a = self(op.a, materialize_idxs=materialize_idxs)
            new_b = self(op.b, materialize_idxs=materialize_idxs)
            return op.record_new(a=new_a, b=new_b)

    @process.register
    def _(self, expr: pyop3.expr.LinearDatBufferExpression, /, *, materialize_idxs):
        if self.index in materialize_idxs:
            assert self._linear

            dat_axes = utils.just_one(pyop3.expr.visitors.get_shape(expr.layout))
            symdat = pyop3.expr.CompositeDat(dat_axes, {dat_axes.leaf_path: expr})
            return pyop3.expr.visitors.materialize_composite_dat(symdat, self.comm, linear=True)
        else:
            with self.enforce_linear():
                new_layout = self(expr.layout, materialize_idxs=materialize_idxs)
            return expr.record_new(layout=new_layout)

    @process.register
    def _(self, dat_expr: pyop3.expr.NonlinearDatBufferExpression, /, **kwargs):
        assert not self._linear
        with self.enforce_linear():
            new_layouts = idict({
                path: self(l, **kwargs)
                for path, l in dat_expr.layouts.items()
            })
        return dat_expr.record_new(layouts=new_layouts)

    @process.register(pyop3.expr.AxisVar)
    @process.register(pyop3.expr.LoopIndexVar)
    @process.register(pyop3.expr.Scalar)
    @process.register(pyop3.expr.ScalarBufferExpression)
    @process.register(pyop3.expr.OpaqueTerminal)
    @process.register(pyop3.expr.NaN)
    def _(self, var, **kwargs):
        return var

    @process.register
    def _(self, mat_expr: pyop3.expr.MatPetscMatBufferExpression, /, *, materialize_idxs):
        assert (self.index, 0) in materialize_idxs
        assert (self.index, 1) in materialize_idxs
        new_row_layout = pyop3.expr.visitors.materialize_composite_dat(
            mat_expr.row_layout, self.comm, linear=False
        )
        new_column_layout = pyop3.expr.visitors.materialize_composite_dat(
            mat_expr.column_layout, self.comm, linear=False
        )
        return mat_expr.record_new(row_layout=new_row_layout, column_layout=new_column_layout)

    @process.register
    def _(self, mat_expr: pyop3.expr.MatArrayBufferExpression, /, **kwargs):
        new_layoutss = []
        layoutss = [mat_expr.row_layouts, mat_expr.column_layouts]
        with self.enforce_linear():
            for layouts in layoutss:
                new_layouts = {
                    path: self(layout, **kwargs)
                    for path, layout in layouts.items()
                }
                new_layoutss.append(idict(new_layouts))
        new_row_layouts, new_column_layouts = new_layoutss
        return mat_expr.record_new(row_layouts=new_row_layouts, column_layouts=new_column_layouts)

    # }}}

    # {{{ pyop3.insn

    @process.register
    def _(self, insn_list: pyop3.insn.InstructionList, /, **kwargs) -> pyop3.insn.InstructionList:
        new_instructions = tuple(
            self(insn, **kwargs) for insn in insn_list
        )
        return insn_list.record_new(instructions=new_instructions)

    @process.register
    def _(self, loop: pyop3.insn.Loop, /, **kwargs):
        new_statements = tuple(self(stmt, **kwargs) for stmt in loop.statements)
        return loop.record_new(statements=new_statements)

    @process.register
    def _(self, assignment: pyop3.insn.NonEmptyArrayAssignment, /, **kwargs):
        new_assignee = self(assignment.assignee, **kwargs)
        new_expression = self(assignment.expression, **kwargs)
        return assignment.record_new(_assignee=new_assignee, _expression=new_expression)

    @process.register(pyop3.insn.StandaloneCalledFunction)
    @process.register(pyop3.insn.Exscan)  # NOTE: not really ideal, relies on not traversing in other visitor
    @process.register(pyop3.insn.NullInstruction)
    def _(self, insn, *args, **kwargs):
        return insn

    # }}}

    # {{{ misc

    @process.register
    def _(self, var: numbers.Number, /, **kwargs):
        return var

    # }}}


def _insert_materialized_indirections(obj: pyop3.obj.Object, materialize_idxs) -> pyop3.insn.Instruction:
    return _MaterializedIndirectionsInserter(obj.comm)(obj, materialize_idxs=materialize_idxs)
