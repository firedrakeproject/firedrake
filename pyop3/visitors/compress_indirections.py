from __future__ import annotations

import contextlib
import functools
import numbers
import types

from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3.axis_tree
import pyop3.expr
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
    # This optimisation is collective but since the array size is part of the
    # heuristic one can get differing optimisation choices on different ranks. We
    # therefore perform all the heuristics on rank 0 and broadcast the selections.
    # TODO: if compress is False I imagine that we don't have to do a bcast here since the
    # result should be the same.
    if insn.comm.rank == 0:
        candidates = collect_candidate_indirections(insn, compress=compress)

        # Find the best cost if we look at each terminal independently. This
        # provides a useful upper bound.
        max_cost = 0
        for terminal_id, terminal_candidates in candidates.items():
            min_terminal_cost = min((cost for cost, _ in terminal_candidates), default=0)
            max_cost += min_terminal_cost
        assert isinstance(max_cost, numbers.Integral)

        # Optimise by dropping any immediately bad candidates
        trimmed_candidates = {}
        for terminal_id, terminal_candidates in candidates.items():
            trimmed_terminal_candidates = []
            min_terminal_cost = min((cost for cost, _ in terminal_candidates))
            for cost, materialize_idxs in terminal_candidates:
                if cost <= max_cost and cost <= min_terminal_cost * MAX_COST_CONSIDERATION_FACTOR:
                    trimmed_terminal_candidates.append((cost, materialize_idxs))
            trimmed_candidates[terminal_id] = trimmed_terminal_candidates
        candidates = trimmed_candidates

        # Optimise the search tree by only considering disjoint subsets of
        # candidates. For example, if we have candidates
        #
        #     {a: [A, B, C, D], b: [X, Y]}
        #
        # we can speed things up by only investigating 4+2 options instead
        # of 4*2.
        # If 'compress' is false we skip this as it introduces unnecessary cost.
        breakpoint()  # intersect materialize idxs?
        disjoint_subsets: list[tuple[dict, set]] = [
            (
                {arg_id: arg_candidates},
                set(ac for ac, _, _ in arg_candidates),
            )
            for arg_id, arg_candidates in candidates.items()
        ]
        if compress:
            # Have to do this repeatedly to ensure subsets are fully disjoint
            while True:
                new_disjoint_subsets = []
                for terminal_id, terminal_candidates in candidates.items():
                    arg_candidate_set = set(ac for ac, _, _ in terminal_candidates)
                    for existing_subset_dict, existing_subset_candidate_set in new_disjoint_subsets:
                        if arg_candidate_set.intersection(existing_subset_candidate_set):
                            existing_subset_dict[terminal_id] = terminal_candidates
                            existing_subset_candidate_set.update(arg_candidate_set)
                            break
                    else:
                        # not found in an existing subset, create a new one
                        subset = ({terminal_id: terminal_candidates}, arg_candidate_set)
                        new_disjoint_subsets.append(subset)

                if new_disjoint_subsets == disjoint_subsets:
                    break

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
        best_candidate = {}
        for candidate_subset, _ in disjoint_subsets:
            # same as above but per subset
            best_subset_candidate = {}
            max_subset_cost = 0
            for terminal_id, terminal_candidates in candidate_subset.items():
                expr, expr_cost, materialize_idxs = min(terminal_candidates, key=lambda item: item[1])
                best_subset_candidate[terminal_id] = (expr, expr_cost, materialize_idxs)
                max_subset_cost += expr_cost

            min_subset_cost = max_subset_cost
            for shared_candidate in utils.expand_collection_of_iterables(candidate_subset):
                cost = 0
                seen_exprs = set()
                for expr, expr_cost, _ in shared_candidate.values():
                    if expr not in seen_exprs:
                        cost += expr_cost
                        seen_exprs.add(expr)

                if cost < min_subset_cost:
                    best_subset_candidate = shared_candidate
                    min_subset_cost = cost
            assert best_subset_candidate is not None
            best_candidate |= best_subset_candidate

        # Identify and broadcast the materialisation indices
        materialize_idxss = {key: idxs for key, (_, _, idxs) in best_candidate.items()}
        insn.comm.bcast(materialize_idxss)

        # Drop cost information from 'best_candidate'
        best_candidate = {key: expr for key, (expr, _, _) in best_candidate.items()}

    else:
        materialize_idxss = insn.comm.bcast(None)
        # identify the dat expressions to materialise using 'materialize_idxss'
        best_candidate = collect_candidate_indirections(insn, compress="anything", selector=idict(materialize_idxss))

    # Materialise any symbolic (composite) dats
    composite_dats = OrderedFrozenSet().union(*map(pyop3.expr.visitors.collect_composite_dats, best_candidate.values()))
    replace_map = {
        comp_dat: pyop3.expr.visitors.materialize_composite_dat(comp_dat, insn.comm)
        for comp_dat in composite_dats
    }
    best_candidate = idict({
        key: pyop3.expr.visitors.replace(expr, replace_map)
        for key, expr in best_candidate.items()
    })

    # Lastly propagate the materialised indirections back through the instruction tree
    return concretize_materialized_indirections(insn, best_candidate)


# TODO: pull out the selector bits, make a separate class
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

    # TODO dont need this any more, just access self.index
    def preprocess_node(self, node) -> tuple[Any, ...]:
        return node, self.index

    @functools.singledispatchmethod
    def process(self, obj: pyop3.obj.Object, /, *args, **kwargs) -> tuple[tuple[Any, int, int], ...]:
        utils.raise_missing_dispatch_handler(obj)

    # {{{ pyop3.expr

    @process.register
    def _(self, op: pyop3.expr.Operator, index, /, *, compress: bool, selector) -> tuple:
        if selector is not None:
            operand_candidatess = tuple(
                self(operand, compress=compress, selector=selector)
                for operand in op.operands
            )

            if not self._collecting:
                return type(op)(*operand_candidatess)
            else:
                if index in selector:
                    op_axes = utils.just_one(get_shape(op))
                    return pyop3.expr.CompositeDat(op_axes, {op_axes.leaf_path: op})
                else:
                    return type(op)(*operand_candidatess)

        if not self._collecting:
            return sum(self(op_, **kwargs) for op_ in op.operands)
            return utils.merge_dicts((self(operand, **kwargs) for operand in op.operands))

        operand_candidatess = tuple(
            self(operand, compress=compress, selector=selector)
            for operand in op.operands
        )

        candidates = []
        for operand_candidates in itertools.product(*operand_candidatess):
            operand_exprs, operand_costs, materialization_indices = zip(*operand_candidates, strict=True)

            materialization_indices = sum(materialization_indices, ())

            # If there is at most one non-zero operand cost then there is no point
            # in compressing the expression.
            if len([cost for cost in operand_costs if cost > 0]) <= 1:
                compress = False

            candidate_expr = type(op)(*operand_exprs)

            # NOTE: This isn't quite correct. For example consider the expression
            # 'mapA[i] + mapA[i]'. The cost is just the cost of 'mapA[i]', not double.
            candidate_cost = sum(operand_costs)
            candidates.append((candidate_expr, candidate_cost, materialization_indices))

        if compress:
            # Now also include a candidate representing the packing of the expression
            # into a Dat. The cost for this is simply the size of the resulting array.
            # Only do this when the cost is large as small arrays will fit in cache
            # and not benefit from the optimisation.
            if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost, _ in candidates):
                op_axes = utils.just_one(get_shape(op))
                op_loop_axes = get_loop_axes(op)
                compressed_expr = pyop3.expr.CompositeDat(op_axes, {op_axes.leaf_path: op})

                op_cost = op_axes.local_size
                for loop_axes in op_loop_axes.values():
                    for loop_axis in loop_axes:
                        op_cost *= loop_axis.component.local_size
                if not isinstance(op_cost, numbers.Integral):
                    raise NotImplementedError("Ragged sizes are not supported")
                candidates.append((compressed_expr, op_cost, (index,)))

        return candidates

    @process.register
    def _(self, expr: pyop3.expr.LinearDatBufferExpression, index, /, *, compress: bool, selector) -> tuple:
        if selector is not None:
            if index in selector:
                return pyop3.expr.CompositeDat(dat_axes, {dat_axes.leaf_path: expr})
            else:
                return expr.record_new(layout=child)

        if not self._collecting:
            with self.collecting():
                return idict({self.index: self(expr.layout, compress=compress, selector=None)})

        breakpoint()

        # The cost of an expression dat (i.e. the memory volume) is given by...
        # Remember that the axes here described the outer loops that exist and that
        # index expressions that do not access data (e.g. 2i+j) have a cost of zero.
        # dat[2i+j] would have a cost equal to ni*nj as those would be the outer loops

        dat_axes = utils.just_one(get_shape(expr.layout))
        dat_loop_axes = get_loop_axes(expr.layout)
        dat_cost = dat_axes.local_size
        for loop_axes in dat_loop_axes.values():
            for loop_axis in loop_axes:
                dat_cost *= loop_axis.component.local_size
        if not isinstance(dat_cost, numbers.Integral):
            raise NotImplementedError("Ragged sizes are not supported")

        child = self(expr.layout, compress=compress,selector=selector)
        candidates = []
        for layout_expr, layout_cost, layout_materialization_indices in child:
            candidate_expr = expr.record_new(layout=layout_expr)

            # TODO: Only apply penalty for non-affine layouts
            candidate_cost = dat_cost + layout_cost * INDIRECTION_PENALTY_FACTOR
            candidates.append((candidate_expr, candidate_cost, layout_materialization_indices))

        if compress:
            if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost, _ in candidates):
                candidates.append((pyop3.expr.CompositeDat(dat_axes, {dat_axes.leaf_path: expr}), dat_cost, (index,)))
        return tuple(candidates)

    @process.register(pyop3.expr.AxisVar)
    @process.register(pyop3.expr.LoopIndexVar)
    @process.register(pyop3.expr.OpaqueTerminal)
    @process.register(pyop3.expr.Scalar)
    @process.register(pyop3.expr.ScalarBufferExpression)
    @process.register(pyop3.expr.NaN)
    def _(self, var: Any, index, /, selector, **kwargs) -> idict:
        if selector is not None:
            assert index not in selector
            return var

        if self._collecting:
            return ()
        else:
            return idict({})
        #
        # else:
        #     # TODO: leave empty? want smallest search space
        #     candidates = [(var, 0, ())]
        #     if self._expect_linear:
        #         return candidates
        #     else:
        #         return {self.index: candidates}



    # @process.register(pyop3.expr.LinearDatBufferExpression)
    # def _(self, dat_expr: pyop3.expr.LinearDatBufferExpression, index, /, *, selector, **kwargs) -> idict:
    #     selector_ = selector[index] if selector is not None else None
    #     return idict({
    #         index: collect_candidate_indirections(dat_expr.layout, selector=selector_, **kwargs)
    #     })
    #
    #
    # @process.register(pyop3.expr.NonlinearDatBufferExpression)
    # def _(self, dat_expr: pyop3.expr.NonlinearDatBufferExpression, index, /, *, selector, **kwargs) -> idict:
    #     candidates = {}
    #     for i, (path, layout) in enumerate(dat_expr.layouts.items()):
    #         selector_ = selector[index, i] if selector is not None else None
    #         candidates[index, i] = self(
    #             layout, selector=selector_, **kwargs
    #         )
    #     return idict(candidates)
    #
    # @process.register(pyop3.expr.MatPetscMatBufferExpression)
    # def _(self, mat_expr: pyop3.expr.MatPetscMatBufferExpression, index, /, *, compress: bool, selector) -> idict:
    #     costs = []
    #     layouts = [mat_expr.row_layout, mat_expr.column_layout]
    #     for i, layout in enumerate(layouts):
    #         cost = loopified_shape(layout)[0].local_size
    #         if not isinstance(cost, numbers.Integral):
    #             raise NotImplementedError("Ragged sizes are not supported")
    #         costs.append(cost)
    #
    #     candidates = {}
    #     if selector is not None:
    #         candidates[index, 0] = mat_expr.row_layout
    #         candidates[index, 1] = mat_expr.column_layout
    #     else:
    #         candidates[index, 0] =  ((mat_expr.row_layout, costs[0], 0),)
    #         candidates[index, 1] =  ((mat_expr.column_layout, costs[1], 0),)
    #     return idict(candidates)
    #
    #
    # # Should be very similar to NonlinearDat case
    # # NOTE: This is a nonlinear type
    # @process.register(pyop3.expr.MatArrayBufferExpression)
    # def _(self, mat_expr: pyop3.expr.MatArrayBufferExpression, index, /, *,  compress: bool, selector) -> idict:
    #     candidates = {}
    #     layoutss = [mat_expr.row_layouts, mat_expr.column_layouts]
    #     for i, layouts in enumerate(layoutss):
    #         for j, (path, layout) in enumerate(layouts.items()):
    #             selector_ = selector[index, i, j] if selector is not None else None
    #             candidates[index, i, j] = self(
    #                 layout, compress=compress, selector=selector_
    #             )
    #     return idict(candidates)



    # }}}

    # {{{ pyop3.insn

    @process.register(pyop3.insn.NullInstruction)
    @process.register(pyop3.insn.Exscan)  # assume we are fine
    def _(self, null: pyop3.insn.InstructionList, index, /, **kwargs) -> idict:
        return idict()


    @process.register(pyop3.insn.InstructionList)
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
    def _(self, terminal: pyop3.insn.NonEmptyTerminal, index, /, *, compress: bool, selector) -> idict:
        if selector is not None:
            candidates = {}
            for i, arg in enumerate(terminal.arguments):
                # drop some of the key
                selector_ = idict({
                    utils.just_one(key[2:]): value
                    for key, value in selector.items()
                    if key[:2] == (index, i)
                })

            per_arg_candidates = self(
                arg, compress=compress, selector=selector_
            )
            for arg_key, value in per_arg_candidates.items():
                candidates[index, i, arg_key] = value
            return idict(candidates)

        candidates = {}
        for i, arg in enumerate(terminal.arguments):
            per_arg_candidates = self(
                arg, compress=compress, selector=None
            )
            candidates |= per_arg_candidates
        return idict(candidates)

    # }}}

    # {{{ misc

    # @process.register(numbers.Number)
    # @process.register(pyop3.expr.AxisVar)
    # @process.register(pyop3.expr.LoopIndexVar)
    # @process.register(pyop3.expr.NaN)
    # @process.register(pyop3.expr.ScalarBufferExpression)
    # def _(self, var: Any, index: int, /, *args, selector, **kwargs) -> tuple[tuple[Any, int, int], ...]:
    #     if selector is not None:
    #         assert index not in selector
    #         return var
    #     else:
    #         return ((var, 0, ()),)

    # }}}


def collect_candidate_indirections(
    obj: pyop3.obj.Object,
    *,
    compress: bool,
    selector=None,
):
    collector = _get_candidate_indirections_collector(obj.comm)
    collector.index = -1  # reset counter
    return collector(obj, compress=compress, selector=selector)


@pyop3.cache.memory_cache(heavy=True)
def _get_candidate_indirections_collector(comm):
    return _CandidateIndirectionsCollector()
