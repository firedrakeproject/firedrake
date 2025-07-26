from __future__ import annotations

import abc
import collections
import functools
import itertools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence, Mapping
from functools import cached_property
from immutabledict import immutabledict as idict
from itertools import chain
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union
from types import GeneratorType

from pyop3.exceptions import Pyop3Exception
import pyrsistent
import pytools

from pyop3 import utils
from pyop3.utils import (
    Id,
    Identified,
    Label,
    Labelled,
    UniqueNameGenerator,
    apply_at,
    as_tuple,
    deprecated,
    flatten,
    has_unique_entries,
    just_one,
    map_when,
    some_but_not_all,
    strictly_all,
    unique,
)


LabelT = Hashable
NodeLabelT = Hashable
ComponentLabelT = Hashable
ComponentRegionLabelT = Hashable
ComponentT = ComponentLabelT  # | ComponentT
PathT = Mapping[NodeLabelT, ComponentLabelT]
ConcretePathT = idict[NodeLabelT, ComponentLabelT]

# ParentT = tuple[PathT, ComponentT] | PathT | | None


class NodeNotFoundException(Exception):
    pass


class EmptyTreeException(Exception):
    pass


class InvalidTreeException(ValueError):
    pass


class TreeMutationException(Pyop3Exception):
    pass


class Node(pytools.ImmutableRecord):
    fields = set()

    def __init__(self):
        pytools.ImmutableRecord.__init__(self)


NodeMapT = Mapping[PathT, Node | None]
ConcreteNodeMapT = idict[ConcretePathT, Node | None]



# TODO delete this class, no longer different tree types
class AbstractTree(abc.ABC):
    def __init__(self, node_map: Mapping[PathT, Node] | None | None = None) -> None:
        self.node_map = as_node_map(node_map)

    def __str__(self) -> str:
        if self.is_empty:
            return "<empty>"
        else:
            return "\n".join(
                self._stringify(path=idict(), begin_prefix="", cont_prefix="")
            )

    def __contains__(self, node) -> bool:
        return self._as_node(node) in self.nodes

    def __bool__(self) -> bool:
        """Return `True` if the tree is non-empty."""
        return not self.is_empty

    @property
    def root(self) -> Node | None:
        return self.node_map.get(idict())

    @property
    def is_empty(self) -> bool:
        assert len(self.node_map) > 0
        return self.node_map == idict({idict(): None})

    @property
    def depth(self) -> int:
        if self.is_empty:
            return 0
        else:
            return postvisit(self, lambda _, *o: max(o or [0]) + 1)

    @cached_property
    def node_ids(self):
        return frozenset(node.id for node in self.nodes)

    @cached_property
    def child_to_parent(self):
        child_to_parent_ = {}
        for parent_id, children in self.node_map.items():
            parent = self._as_node(parent_id)
            for i, child in enumerate(children):
                child_to_parent_[child] = (parent, i)
        return child_to_parent_

    @cached_property
    def nodes(self) -> tuple[Node]:
        return tuple(filter(None, self.node_map.values()))

    # @property
    # @abc.abstractmethod
    # def leaves(self):
    #     """Return the leaves of the tree."""
    #     pass
    #
    # @property
    # def leaf(self):
    #     return just_one(self.leaves)

    def is_leaf(self, node):
        return self._as_node(node) in self.leaves

    def parent(self, node):
        node = self._as_node(node)
        return self.child_to_parent[node]

    def children(self, path: PathT) -> tuple[Node | None]:
        """"Return the child nodes from a path.

        If the path points to a leaf then the children may include `None`.

        """
        path = as_path(path)

        children_ = []
        node = self.node_map[path]
        for component_label in node.component_labels:
            child_path = path | {node.label: component_label}
            child = self.node_map[child_path]
            children.append(child)
        return tuple(children)

    @staticmethod
    def _parse_node(node):
        if isinstance(node, Node):
            return node
        else:
            raise TypeError(f"No handler defined for {type(node).__name__}")

    def _stringify(
        self,
        *,
        path: ConcretePathT,
        begin_prefix: str,
        cont_prefix: str,
    ) -> tuple[str]:
        assert not self.is_empty

        node = self.node_map[path]
        nodestr = [f"{begin_prefix}{node}"]
        for i, component_label in enumerate(node.component_labels):
            path_ = path | {node.label: component_label}

            last_child = i == len(node.component_labels) - 1
            next_begin_prefix = f"{cont_prefix}{'└' if last_child else '├'}──➤ "
            next_cont_prefix = f"{cont_prefix}{' ' if last_child else '│'}    "
            if self.node_map[path_]:
                nodestr += self._stringify(
                    path=path_, begin_prefix=next_begin_prefix, cont_prefix=next_cont_prefix
                )

        return tuple(nodestr)

    def _as_node(self, node):
        if node is None:
            return None
        else:
            return node if isinstance(node, Node) else self.id_to_node[node]

    @staticmethod
    def _as_node_id(node):
        return node.id if isinstance(node, Node) else node


class LabelledNodeComponent(pytools.ImmutableRecord):
    fields = {"label"}

    def __init__(self, label=None):
        self.label = label


class MultiComponentLabelledNode(Node, Labelled):
    fields = Node.fields | {"label"}

    def __init__(self, label=None):
        Node.__init__(self)
        Labelled.__init__(self, label)

        if not utils.has_unique_entries(self.component_labels):
            raise ValueError("Duplicate component labels found")

    @property
    def degree(self) -> int:
        return len(self.component_labels)

    @property
    @abc.abstractmethod
    def component_labels(self) -> tuple:
        pass

    @property
    def component_label(self):
        return just_one(self.component_labels)


class LabelledTree(AbstractTree):

    # {{{ abstract methods

    @classmethod
    @abc.abstractmethod
    def as_node(self, obj: Any) -> Node:
        """Convert an object into a tree node."""

    # }}}

    # {{{ constructors

    @classmethod
    def from_iterable(cls, iterable: Iterable) -> LabelledTree:
        if not iterable:
            return cls()

        node_map = {}
        path = idict()
        for node in iterable:
            node = cls.as_node(node)
            node_map.update({path: node})
            path = path | {node.label: node.component_label}
        return cls(node_map)

    @classmethod
    def from_nest(cls, nest: Mapping[Node, Sequence[Mapping | Node]] | Node) -> LabelledTree:
        if isinstance(nest, Node):
            return cls(nest)
        else:
            node_map = cls._node_map_from_nest(nest=nest, path=idict())
            return cls(node_map)

    @classmethod
    def _node_map_from_nest(cls, *, nest: Mapping[Node, Sequence[Mapping | Node]], path: ConcretePathT) -> ConcretePathT:
        if len(nest) > 1:
            raise InvalidTreeException(
                "Nest contains multiple nodes at the same level"
            )

        node, subnests = utils.just_one(nest.items())
        node = cls.as_node(node)

        if isinstance(subnests, Node) and node.degree == 1:
            subnests = (subnests,)

        node_map = {path: node}
        for component_label, subnest in zip(node.component_labels, subnests, strict=True):
            path_ = path | {node.label: component_label}

            if isinstance(subnest, Mapping):
                sub_node_map = cls._node_map_from_nest(nest=subnest, path=path_)
            else:
                sub_node_map = {path_: subnest}
            node_map |= sub_node_map
        return idict(node_map)

    # }}}

    def __init__(self, node_map=None):
        super().__init__(node_map=node_map)
        self._cache = {}

    def __eq__(self, other):
        return type(other) is type(self) and other._hash_key == self._hash_key

    def __hash__(self):
        return hash(self._hash_key)

    @cached_property
    def _hash_key(self):
        return (self._hash_node_map,)

    @cached_property
    def _hash_node_map(self):
        if self.is_empty:
            return idict()

        counter = itertools.count()
        return self._collect_hash_node_map(None, None, counter)

    def _collect_hash_node_map(self, old_parent_id, new_parent_id, counter):
        if old_parent_id not in self.node_map:
            return idict()

        nodes = []
        node_map = {}
        for old_node in self.node_map[old_parent_id]:
            if old_node is not None:
                new_node = old_node.copy(id=f"id_{next(counter)}")
                node_map.update(self._collect_hash_node_map(old_node.id, new_node.id, counter))
            else:
                new_node = None

            nodes.append(new_node)

        node_map[new_parent_id] = freeze(nodes)
        return freeze(node_map)

    def child(self, parent: ParentT) -> Node:
        assert False, "old code, just use node_map"
        """Return the child node of ``parent``.

        Parameters
        ----------
        parent :
            The parent node specification. This can be a 2-tuple of parent
            node and component (label) or `None`, which will return the
            root node.

        Returns
        -------
        child : Node
            The child node.

        """
        if parent is None:
            return self.root

        return self.node_map[parent]

        path, component = parent
        parent_node = self.node_map[path]
        # full_path = 
        # return utils.just_one(self.node_map[path].components
        component_label = as_component_label(parent_component)
        component_index = parent_node.component_labels.index(component_label)
        return children[component_index]

    @cached_property
    def leaves(self) -> tuple[Node]:
        return tuple(self.node_map[parent_path(leaf_path)] for leaf_path in self.leaf_paths)

    # # TODO: Alternatively might be nicer to return just the nodes. The components are obvious
    # @cached_property
    # def leaves(self) -> tuple[tuple[Node, ComponentLabelT]]:
    #     """Return the leaves of the tree."""
    #     if self.is_empty:
    #         raise ValueError("Error here? Not an intuitive return type")
    #
    #     return self._collect_leaves(path=idict())
    #
    # def _collect_leaves(self, *, path: PathT) -> tuple[tuple[Node, ComponentLabelT]]:
    #     leaves = []
    #     node = self.node_map[path]
    #     for component_label in node.component_labels:
    #         path_ = path | {node.label: component_label}
    #         if self.node_map[path_]:
    #             leaves.extend(self._collect_leaves(path=path_))
    #         else:
    #             leaves.append((node, component_label))
    #     return tuple(leaves)

    @property
    def is_linear(self) -> bool:
        return len(self.leaf_paths) == 1

    def _uniquify_node_labels(self, node_map, node=None, seen_labels=None):
        if not node_map:
            return

        if node is None:
            node = just_one(node_map[None])
            seen_labels = frozenset({node.label})

        for i, subnode in enumerate(node_map.get(node.id, [])):
            if subnode is None:
                continue
            if subnode.label in seen_labels:
                new_label = UniqueNameGenerator(set(seen_labels))(subnode.label)
                assert new_label not in seen_labels
                subnode = subnode.copy(label=new_label)
                node_map[node.id][i] = subnode
            self._uniquify_node_labels(node_map, subnode, seen_labels | {subnode.label})

    # do as a traversal since there is an ordering constraint in how we replace IDs
    def _uniquify_node_ids(self, node_map, existing_ids, node=None):
        assert False, "old code"
        if not node_map:
            return

        node_id = node.id if node is not None else None
        for i, subnode in enumerate(node_map.get(node_id, [])):
            if subnode is None:
                continue
            if subnode.id in existing_ids:
                new_id = subnode.unique_id()
                assert new_id not in existing_ids
                existing_ids.add(new_id)
                new_subnode = subnode.copy(id=new_id)
                node_map[node_id][i] = new_subnode
                node_map[new_id] = node_map.pop(subnode.id)
                self._uniquify_node_ids(node_map, existing_ids, new_subnode)

    def visited_nodes(self, path: PathT) -> tuple[tuple[Node, ComponentLabelT], ...]:
        path = as_path(path)

        ordered_path = utils.just_one(
            path_
            for path_ in self.node_map
            if path_ == path
        )

        nodes = []
        for path_acc in accumulate_path(ordered_path, skip_last=True):
            node = self.node_map[path_acc]
            # NOTE: this is kind of obvious
            component_label = path[node.label]
            nodes.append((node, component_label))
        return tuple(nodes)

    @cached_property
    def _paths(self):
        assert False, "old code"
        def paths_fn(node, component_label, current_path):
            if current_path is None:
                current_path = ()
            new_path = current_path + ((node.label, component_label),)
            paths_[node.id, component_label] = new_path
            return new_path

        paths_ = {}
        previsit(self, paths_fn)
        return idict(paths_)

    # TODO interface choice about whether we want whole nodes, ids or labels in paths
    # maybe need to distinguish between paths, ancestors and label-only?
    @cached_property
    def _paths_with_nodes(self):
        def paths_fn(node, component_label, current_path):
            if current_path is None:
                current_path = ()
            new_path = current_path + ((node, component_label),)
            paths_[node.id, component_label] = new_path
            return new_path

        paths_ = {}
        previsit(self, paths_fn)
        return idict(paths_)

    def ancestors(self, node, component_label):
        """Return the ancestors of a ``(node_id, component_label)`` 2-tuple."""
        return idict(
            {
                nd: cpt
                for nd, cpt in self.path(node, component_label).items()
                if nd != node.label
            }
        )

    def path(self, target: tuple[Node, ComponentT] | None) -> idict:
        """Return the path to ``target``."""
        assert False, "old code that is no longer valid as nodes can crop up in multiple paths"
        if target is None:
            return idict()

        node, component = target
        component_label = as_component_label(component)
        path_ = self._paths[node_id, component_label]
        if ordered:
            return path_
        else:
            return idict(path_)

    def path_with_nodes(
        self, node, component_label=None, ordered=False, and_components=False
    ) -> idict:
        assert False, "old code"
        if node is None:
            return idict()

        # TODO: make target always be a 2-tuple
        if isinstance(node, tuple):
            assert component_label is None
            node, component_label = node

        component_label = as_component_label(component_label)
        node_id = self._as_node_id(node)
        path_ = self._paths_with_nodes[node_id, component_label]
        if and_components:
            path_ = tuple(
                (ax, just_one(cpt for cpt in ax.components if cpt.label == clabel))
                for ax, clabel in path_
            )
        if ordered:
            return path_
        else:
            return idict(path_)

    @cached_property
    def paths(self) -> tuple[idict, ...]:
        """Return all possible paths through the tree."""
        return self.node_map.keys()

    @cached_property
    def leaf_paths(self) -> tuple[ConcretePathT, ...]:
        """Return the paths to each leaf of the tree."""
        return tuple(path for path, node in self.node_map.items() if node is None)

    @property
    def leaf_path(self) -> idict:
        return just_one(self.leaf_paths)

    @cached_property
    def ordered_leaf_paths(self):
        assert False, "use leaf_paths instead"
        return tuple(self.path(*leaf, ordered=True) for leaf in self.leaves)

    @cached_property
    def leaf_node_paths(self):
        return tuple(self.path_with_nodes(*leaf) for leaf in self.leaves)

    @cached_property
    def ordered_leaf_node_paths(self):
        return tuple(self.path_with_nodes(*leaf, ordered=True) for leaf in self.leaves)

    def _node_from_path(self, path):
        if not path:
            return None

        path_ = dict(path)
        node = self.root
        while True:
            cpt_label = path_.pop(node.label)
            cpt_index = node.component_labels.index(cpt_label)
            new_node = self.node_map.get(node.id, [None] * node.degree)[
                cpt_index
            ]

            # if we are a leaf then return the final bit
            if path_:
                node = new_node
            else:
                return node, node.components[cpt_index]
        assert False, "shouldn't get this far"

    # bad name
    def detailed_path(self, path):
        node = self._node_from_path(path)
        if node is None:
            return idict()
        else:
            return self.path_with_nodes(*node, and_components=True)

    def is_valid_path(self, path, complete=True, leaf=False):
        assert False, "old code"
        if leaf:
            all_paths = [set(self.path(node, cpt).items()) for node, cpt in self.leaves]
        else:
            all_paths = [
                set(self.path(node, cpt).items())
                for node in self.nodes
                for cpt in node.components
            ]
            all_paths.append(set())  # handle empty case

        path_set = set(path.items())

        compare = operator.eq if complete else operator.le

        for path_ in all_paths:
            if compare(path_set, path_):
                return True
        return False

    @cached_property
    def node_labels(self):
        return frozenset(n.label for n in self.nodes)

    def find_component(self, node_label, cpt_label, also_node=False):
        """Return the first component in the tree matching the given labels.

        Notes
        -----
        This will return the first component matching the labels. Multiple may exist
        but we assume that they are identical.

        """
        for node in self.nodes:
            if node.label == node_label:
                for cpt in node.components:
                    if cpt.label == cpt_label:
                        if also_node:
                            return node, cpt
                        else:
                            return cpt
        raise ValueError("Matching component not found")

    def _relabel_node_map(self, replace_map: Mapping) -> Mapping:
        new_node_map = {}
        for parent_id, children in self.node_map.items():
            new_children = []
            for child in children:
                if child is not None:
                    new_child = child.copy(label=replace_map.get(child.label, child.label))
                else:
                    new_child = None
                new_children.append(new_child)
            new_node_map[parent_id] = new_children
        return new_node_map

    # TODO, could be improved, same as other Tree apart from [None, None, ...] bit
    @staticmethod
    def _parse_parent_to_children(node_map) -> idict:
        if not node_map:
            return idict()

        if isinstance(node_map, Node):
            # just passing root
            node_map = {None: (node_map,)}
        else:
            node_map = dict(node_map)

        if None not in node_map:
            raise ValueError("Root missing from tree")
        if len(node_map[None]) != 1:
            raise ValueError("Multiple roots provided, this is not allowed")

        nodes = [
            node
            for node in chain.from_iterable(node_map.values())
            if node is not None
        ]
        if any(
            parent not in nodes
            for parent in node_map.keys() - {None}
        ):
            raise ValueError("Tree is disconnected")
        for node in nodes:
            if node not in node_map.keys():
                node_map[node] = [None] * node.degree
        return idict(node_map)

    @staticmethod
    def _parse_node(node):
        if isinstance(node, MultiComponentLabelledNode):
            return node
        else:
            raise TypeError(f"No handler defined for {type(node).__name__}")


class MutableLabelledTreeMixin:
    def add_node(self, path: PathT, node: Node) -> MutableLabelledTreeMixin:
        """Return a new tree with ``node`` attached at ``path``."""
        path = as_path(path)

        if self.node_map[path]:
            raise TreeMutationException(
                "A node already exists at this location."
            )

        if self.is_empty:
            return type(self)(node)

        *parent_path, (parent_axis_label, parent_component_label) = path.items()
        parent_path = as_path(parent_path)
        try:
            parent_node = self.node_map[parent_path]
        except KeyError:
            raise TreeMutationException("Parent node does not exist")
        if parent_axis_label != parent_node.label or parent_component_label not in parent_node.component_labels:
            raise TreeMutationException("Bad parent descriptor")

        return type(self)(self.node_map | {path: node})

    def add_subtree(self, path: PathT, subtree: LabelledTree) -> MutableLabelledTreeMixin:
        """Attach another tree to a leaf of the current tree."""
        path = as_path(path)

        if path not in self.leaf_paths:
            raise TreeMutationException("Can only attach a subtree to an existing leaf")

        if subtree.is_empty:
            return self

        # TODO: breaks abstraction
        from pyop3.axtree.tree import _UnitAxisTree
        if isinstance(subtree, _UnitAxisTree):
            return self

        node_map = dict(self.node_map)
        for subpath, subnode in subtree.node_map.items():
            assert not (path.keys() & subpath.keys())
            node_map[path | subpath] = subnode
        return type(self)(node_map)

    def subtree(self, path: PathT) -> MutableLabelledTreeMixin:
        """Return the subtree with ``path`` as the root."""
        path = as_path(path)

        if path not in self.node_map:
            raise TreeMutationException("Provided path does not exist in the tree")

        trimmed_node_map = {}
        path_set = frozenset(path.items())
        for orig_path, node in self.node_map.items():
            orig_path_set = frozenset(orig_path.items())
            if path_set <= orig_path_set:
                trimmed_path = idict(
                    (axis_label, component_label)
                    for axis_label, component_label in orig_path.items()
                    if (axis_label, component_label) not in path.items()
                )
                trimmed_node_map[trimmed_path] = node
        return type(self)(trimmed_node_map)

    def drop_subtree(self, path: PathT, *, allow_empty_subtree=False) -> MutableLabelledTreeMixin:
        path = as_path(path)

        if path not in self.node_map:
            raise TreeMutationException("Provided path does not exist in the tree")

        if path in self.leaf_paths:
            if allow_empty_subtree:
                return self
            # ie dropping nothing is probably unexpected behaviour
            else:
                assert False

        subtree = self.subtree(path)

        trimmed_node_map = {}
        for orig_path, node in self.node_map.items():
            if node in subtree.node_map.values():
                continue
            trimmed_node_map[orig_path] = node
        return type(self)(trimmed_node_map)

    def relabel(self, labels: Mapping):
        assert False, "old code"
        node_map = self._relabel_node_map(labels)
        return type(self)(node_map)


def as_component_label(component):
    if isinstance(component, LabelledNodeComponent):
        return component.label
    else:
        return component


def previsit(
    tree,
    fn,
    current_node: Optional[Node] = None,
    prev=None,
) -> Any:
    if tree.is_empty:
        raise RuntimeError("Cannot traverse an empty tree")

    current_node = current_node or tree.root
    for cpt_label in current_node.component_labels:
        next = fn(current_node, cpt_label, prev)
        if subnode := tree.child(current_node, cpt_label):
            previsit(tree, fn, subnode, next)


def postvisit(tree, fn, path: PathT = idict(), **kwargs) -> Any:
    """Traverse the tree in postorder.

    # TODO rewrite
    Parameters
    ----------
    tree: Tree
        The tree to be visited.
    fn: function(node, *fn_children)
        A function to be applied at each node. The function should take
        the node to be visited as its first argument, and the results of
        visiting its children as any further arguments.
    """
    if tree.is_empty:
        raise RuntimeError("Cannot traverse an empty tree")

    node = tree.node_map[path]

    child_results = []
    for component_label in node.component_labels:
        path_ = path | {node.label: component_label}

        if tree.node_map[path_]:
            child_result = postvisit(tree, fn, path_, **kwargs)
            child_results.append(child_result)

    return fn(node, *child_results, **kwargs)


def as_node_map(node_map: Any) -> idict:
    node_map = _as_node_map(node_map)
    return fixup_node_map(node_map)


@functools.singledispatch
def _as_node_map(obj: Any, /) -> idict:
    if obj is None:
        return idict()
    else:
        raise TypeError(f"No handler provided for {type(obj).__name__}")


@_as_node_map.register(Mapping)
def _(node_map: Mapping, /) -> idict:
    return idict(node_map)


@_as_node_map.register(Node)
def _(node: Node, /) -> idict:
    return idict({idict(): node})


def fixup_node_map(node_map: NodeMapT) -> ConcreteNodeMapT:
    unvisited = dict(node_map)
    complete_node_map = _fixup_node_map(path=idict(), unvisited=unvisited)

    if unvisited:
        raise InvalidTreeException("There are orphaned entries in the node map")

    return complete_node_map

def _fixup_node_map(*, path: idict, unvisited: dict) -> ConcreteNodeMapT:
    if path not in unvisited:
        # at a leaf, attach a 'None'
        return idict({path: None})

    node = unvisited.pop(path)

    if node is None:
        # at a leaf, attach a 'None'
        return idict({path: None})

    if node.label in path.keys():
        raise InvalidTreeException(f"Duplicate label '{node.label}' found along a path")

    node_map = {path: node}
    for component_label in node.component_labels:
        path_ = path | {node.label: component_label}
        node_map |= _fixup_node_map(path=path_, unvisited=unvisited)
    return idict(node_map)


@functools.singledispatch
def as_path(obj: Any) -> ConcretePathT:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@as_path.register(idict)
def _(path: idict) -> ConcretePathT:
    return path


@as_path.register(Iterable)
def _(path: Iterable) -> ConcretePathT:
    return idict(path)


def parent_path(path: PathT) -> ConcretePathT:
    return idict({
        node_label: component_label
        for node_label, component_label in list(path.items())[:-1]
    })


def accumulate_path(path: PathT, *, skip_last: bool = False) -> tuple[ConcretePathT, ...]:
    path_acc = idict()
    paths = [path_acc]
    for node_label, component_label in path.items():
        path_acc = path_acc | {node_label: component_label}
        paths.append(path_acc)

    if skip_last:
        paths = paths[:-1]

    return tuple(paths)


def filter_path(orig_path: PathT, to_remove: PathT) -> ConcretePathT:
    orig_path = as_path(orig_path)
    to_remove = as_path(to_remove)

    filtered_path = {}
    for node_label, component_label in orig_path.items():
        if (node_label, component_label) not in to_remove.items():
            filtered_path[node_label] = component_label
    return idict(filtered_path)
