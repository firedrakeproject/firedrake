from functools import cached_property

from immutabledict import immutabledict as idict

import pyop3.index_tree.tree
import pyop3.record

from .tree import LoopContextAwareAxisTreeLike


@pyop3.record.frozenrecord()
class LoopContextSensitiveAxisTreeLike(
    LoopContextAwareAxisTreeLike,
    pyop3.index_tree.tree.LoopContextSensitive,
):

    # {{{ instance attrs

    trees: idict  # context to tree

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        trees_key = {}
        for path, tree in self.trees.items():
            trees_key[visitor.relabel_axis_tree_path(path)] = visitor(tree)
        trees_key = idict(trees_key)
        return (type(self), trees_key)

    def __init__(self, trees: Mapping):
        trees=idict(trees)
        object.__setattr__(self, "trees", trees)

    # }}}

    @property
    def context_map(self):  # old alias
        return self.trees

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.context_map!r})"

    def __str__(self) -> str:
        return "\n".join(
            f"{context}\n{tree}" for context, tree in self.context_map.items()
        )

    def __getitem__(self, indices) -> ContextSensitiveAxisTree:
        raise NotImplementedError
        # TODO think harder about composing context maps
        # answer is something like:
        # new_context_map = {}
        # for context, axes in self.context_map.items():
        #     for context_, axes_ in index_axes(axes, indices).items():
        #         new_context_map[context | context_] = axes_
        # return ContextSensitiveAxisTree(new_context_map)

    def index(self) -> LoopIndex:
        from pyop3.index_tree import LoopIndex

        return LoopIndex(self)

    @cached_property
    def datamap(self):
        return merge_dicts(axes.datamap for axes in self.context_map.values())

    @cached_property
    def sf(self):
        return single_valued([ax.sf for ax in self.context_map.values()])

    @cached_property
    def unindexed(self):
        return single_valued([ax.unindexed for ax in self.context_map.values()])

    @cached_property
    def context_free(self):
        return just_one(self.context_map.values())


