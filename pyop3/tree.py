from __future__ import annotations

import abc
import collections
import functools
import itertools
import operator
from collections import defaultdict
from collections.abc import Hashable, Sequence, Mapping
from functools import cached_property
from immutabledict import immutabledict
from itertools import chain
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

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


class NodeNotFoundException(Exception):
    pass


class EmptyTreeException(Exception):
    pass


class InvalidTreeException(ValueError):
    pass


LabelT = Hashable
NodeLabelT = Hashable
ComponentLabelT = Hashable
ComponentT = ComponentLabelT  # | ComponentT
PathT = Mapping[NodeLabelT, ComponentLabelT]
ConcretePathT = immutabledict[NodeLabelT, ComponentLabelT]

# ParentT = tuple[PathT, ComponentT] | PathT | | None


class Node(pytools.ImmutableRecord):
    fields = set()

    def __init__(self):
        pytools.ImmutableRecord.__init__(self)


# TODO delete this class, no longer different tree types
class AbstractTree(abc.ABC):
    def __init__(self, node_map: Mapping[PathT, Node] | None | None = None) -> None:
        self.node_map = as_node_map(node_map)

    def __str__(self):
        return self._stringify()

    def __contains__(self, node) -> bool:
        return self._as_node(node) in self.nodes

    def __bool__(self) -> bool:
        """Return `True` if the tree is non-empty."""
        return not self.is_empty

    @property
    def root(self) -> Node | None:
        return self.node_map.get(immutabledict())

    @property
    def is_empty(self) -> bool:
        return not self.node_map

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
    def id_to_node(self):
        return freeze({node.id: node for node in self.nodes})

    @cached_property
    def nodes(self):
        # NOTE: Keep this sorted! Else strange results occur
        if self.is_empty:
            return ()
        return self._collect_nodes(self.root)

    def _collect_nodes(self, node):
        assert not self.is_empty
        nodes = [node]
        for subnode in self.children(node):
            if subnode is None:
                continue
            nodes.extend(self._collect_nodes(subnode))
        return tuple(nodes)

    @property
    @abc.abstractmethod
    def leaves(self):
        """Return the leaves of the tree."""
        pass

    @property
    def leaf(self):
        return just_one(self.leaves)

    def is_leaf(self, node):
        return self._as_node(node) in self.leaves

    def parent(self, node):
        node = self._as_node(node)
        return self.child_to_parent[node]

    def children(self, node: Node) -> tuple[Node | None]:
        """"Return the child nodes of ``node``."""
        return self.node_map.get(node, ())

    @staticmethod
    def _parse_node(node):
        if isinstance(node, Node):
            return node
        else:
            raise TypeError(f"No handler defined for {type(node).__name__}")

    def _stringify(
        self,
        node=None,
        begin_prefix="",
        cont_prefix="",
    ):
        if self.is_empty:
            return "<empty>"

        node = node or self.root

        nodestr = [f"{begin_prefix}{node}"]
        children = self.children(node)
        for i, child in enumerate(children):
            last_child = i == len(children) - 1
            next_begin_prefix = f"{cont_prefix}{'└' if last_child else '├'}──➤ "
            next_cont_prefix = f"{cont_prefix}{' ' if last_child else '│'}    "
            if child is not None:
                nodestr += self._stringify(child, next_begin_prefix, next_cont_prefix)

        if not strictly_all([begin_prefix, cont_prefix]):
            return "\n".join(nodestr)
        else:
            return nodestr

    def _as_node(self, node):
        if node is None:
            return None
        else:
            return node if isinstance(node, Node) else self.id_to_node[node]

    @staticmethod
    def _as_node_id(node):
        return node.id if isinstance(node, Node) else node


class LabelledNodeComponent(pytools.ImmutableRecord, Labelled):
    fields = {"label"}

    def __init__(self, label=None):
        pytools.ImmutableRecord.__init__(self)
        Labelled.__init__(self, label)


class MultiComponentLabelledNode(Node, Labelled):
    fields = Node.fields | {"label"}

    def __init__(self, label=None):
        Node.__init__(self)
        Labelled.__init__(self, label)

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
            return immutabledict()

        counter = itertools.count()
        return self._collect_hash_node_map(None, None, counter)

    def _collect_hash_node_map(self, old_parent_id, new_parent_id, counter):
        if old_parent_id not in self.node_map:
            return immutabledict()

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

    # TODO: Could this return (node, component) instead of (node, component_label)?
    # TODO: Alternatively might be nicer to return just the nodes. The components are obvious
    @cached_property
    def leaves(self) -> tuple[tuple[Node, ComponentLabelT]]:
        """Return the leaves of the tree."""
        if self.is_empty:
            raise ValueError("Error here? Not an intuitive return type")

        return self._collect_leaves(path=immutabledict())

    def _collect_leaves(self, *, path: PathT) -> tuple[tuple[Node, ComponentLabelT]]:
        leaves = []
        node = self.node_map[path]
        for component_label in node.component_labels:
            path_ = path | {node.label: component_label}
            if path_ in self.node_map:
                leaves.extend(self._collect_leaves(path_))
            else:
                leaves.append((node, component_label))
        return tuple(leaves)

    @property
    def is_linear(self) -> bool:
        return len(self.leaves) == 1

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
        return immutabledict(paths_)

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
        return immutabledict(paths_)

    def ancestors(self, node, component_label):
        """Return the ancestors of a ``(node_id, component_label)`` 2-tuple."""
        return immutabledict(
            {
                nd: cpt
                for nd, cpt in self.path(node, component_label).items()
                if nd != node.label
            }
        )

    def path(self, target: tuple[Node, ComponentT] | None) -> immutabledict:
        """Return the path to ``target``."""
        assert False, "old code that is no longer valid as nodes can crop up in multiple paths"
        if target is None:
            return immutabledict()

        node, component = target
        component_label = as_component_label(component)
        path_ = self._paths[node_id, component_label]
        if ordered:
            return path_
        else:
            return immutabledict(path_)

    def path_with_nodes(
        self, node, component_label=None, ordered=False, and_components=False
    ) -> immutabledict:
        if node is None:
            return immutabledict()

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
            return immutabledict(path_)

    @cached_property
    def paths(self) -> tuple[immutabledict, ...]:
        """Return all possible paths through the tree."""
        return self.node_map.keys()

    @cached_property
    def leaf_paths(self) -> tuple[PathT, ...]:
        """Return the paths to each leaf of the tree."""
        if self.is_empty:
            return (immutabledict(),)
        else:
            return self._collect_leaf_paths(path=immutabledict())

    def _collect_leaf_paths(self, *, path: ConcretePathT) -> tuple[ConcretePathT, ...]:
        leaf_paths = []
        node = self.node_map[path]
        for component_label in node.component_labels:
            path_ = path | {node.label: component_label}
            if path_ not in self.node_map:
                leaf_paths.append(path_)
            else:
                leaf_paths.extend(
                    self._collect_leaf_paths(path=path_)
                )
        return tuple(leaf_paths)

    @property
    def leaf_path(self) -> immutabledict:
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
            return immutabledict()
        else:
            return self.path_with_nodes(*node, and_components=True)

    def is_valid_path(self, path, complete=True, leaf=False):
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

    @classmethod
    def _from_nest(cls, nest):
        # TODO add appropriate exception classes
        if isinstance(nest, collections.abc.Mapping):
            assert len(nest) == 1
            node, subnodes = just_one(nest.items())
            node = cls._parse_node(node)

            if isinstance(subnodes, collections.abc.Mapping):
                if len(subnodes) == 1 and isinstance(
                    just_one(subnodes.keys()), MultiComponentLabelledNode
                ):
                    # just one subnode
                    cidxs = [0]
                    subnodes = [subnodes]
                else:
                    # mapping of component labels to subnodes
                    cidxs = [
                        node.component_labels.index(clabel)
                        for clabel in subnodes.keys()
                    ]
                    subnodes = subnodes.values()
            elif isinstance(subnodes, collections.abc.Sequence):
                cidxs = range(node.degree)
            else:
                if node.degree != 1:
                    raise ValueError
                cidxs = [0]
                subnodes = [subnodes]

            children = [None] * node.degree
            parent_to_children = {}
            for cidx, subnode in zip(cidxs, subnodes, strict=True):
                subnode_, sub_p2c = cls._from_nest(subnode)
                children[cidx] = subnode_
                parent_to_children.update(sub_p2c)
            parent_to_children[node.id] = children
            return node, parent_to_children
        else:
            node = cls._parse_node(nest)
            return node, {node.id: [None] * node.degree}

    # TODO, could be improved, same as other Tree apart from [None, None, ...] bit
    @staticmethod
    def _parse_parent_to_children(node_map) -> immutabledict:
        if not node_map:
            return immutabledict()

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
        return immutabledict(node_map)

    @staticmethod
    def _parse_node(node):
        if isinstance(node, MultiComponentLabelledNode):
            return node
        else:
            raise TypeError(f"No handler defined for {type(node).__name__}")


class TreeMutationException(Pyop3Exception):
    pass


class MutableLabelledTreeMixin:
    def add_node(self, path: PathT, node: Node) -> MutableLabelledTreeMixin:
        """Return a new tree with ``node`` attached at ``path``."""
        if path in self.node_map:
            raise TreeMutationException(
                "A node already exists at this location."
            )

        if self.is_empty:
            return type(self)(node)

        *parent_path, (parent_axis_label, parent_component_label) = path.items()

        breakpoint()  # check valid!

        return type(self)(self.node_map | {path: node})

    def add_subtree(
        self,
        subtree,
        parent=None,
        component=None,
        *,
        uniquify: bool = False,
        uniquify_ids=False,
    ):
        """
        Parameters
        ----------
        etc
            ...
        uniquify
            If ``False``, duplicate ``ids`` between the tree and subtree
            will raise an exception. If ``True``, the ``ids`` will be changed
            to avoid the clash.
            Also fixes node labels.

        """
        # FIXME bad API, uniquify implies uniquify labels only
        # There are cases where the labels should be distinct but IDs may clash
        # e.g. adding subaxes for a matrix
        if uniquify_ids:
            assert not uniquify

        if uniquify:
            uniquify_ids = True

        if parent is None:
            assert component is None, "makes no sense otherwise"
        elif isinstance(parent, tuple):  # improved API
            assert component is None
            parent, component = parent
        else:
            pass

        if not parent:
            raise NotImplementedError("TODO")

        if subtree.is_empty:
            return self

        assert isinstance(parent, MultiComponentLabelledNode)
        clabel = as_component_label(component)
        cidx = parent.component_labels.index(clabel)
        parent_to_children = {p: list(ch) for p, ch in self.node_map.items()}

        sub_p2c = {p: list(ch) for p, ch in subtree.node_map.items()}
        if uniquify_ids:
            self._uniquify_node_ids(sub_p2c, set(parent_to_children.keys()))
            assert (
                len(set(sub_p2c.keys()) & set(parent_to_children.keys()) - {None}) == 0
            )

        subroot = just_one(sub_p2c.pop(None))
        parent_to_children[parent.id][cidx] = subroot
        parent_to_children.update(sub_p2c)

        if uniquify:
            self._uniquify_node_labels(parent_to_children)

        return type(self)(parent_to_children)

    def subtree(self, axis):
        """Return the subtree with ``axis`` as the root."""
        node_map = {None: (axis,)}
        node_map.update(self._collect_tree(axis))
        return type(self)(node_map)

    def _collect_tree(self, axis) -> dict:
        node_map = {axis.id: self.node_map[axis.id]}
        for component in axis.components:
            if subaxis := self.child(axis, component):
                subnode_map = self._collect_tree(subaxis)
                node_map.update(subnode_map)
        return node_map

    def relabel(self, labels: Mapping):
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


def postvisit(tree, fn, current_node: Optional[Node] = None, **kwargs) -> Any:
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

    current_node = current_node or tree.root
    return fn(
        current_node,
        *(
            postvisit(tree, fn, child, **kwargs)
            for child in filter(None, tree.children(current_node))
        ),
        **kwargs,
    )


def as_node_map(node_map: Any) -> immutabledict:
    node_map = _as_node_map(node_map)
    check_node_map(node_map)
    return node_map


@functools.singledispatch
def _as_node_map(obj: Any, /) -> immutabledict:
    if obj is None:
        return immutabledict()
    else:
        raise TypeError(f"No handler provided for {type(obj).__name__}")


@_as_node_map.register(Mapping)
def _(node_map: Mapping, /) -> immutabledict:
    return immutabledict(node_map)


@_as_node_map.register(Node)
def _(node: Node, /) -> immutabledict:
    return immutabledict({immutabledict(): node})


def check_node_map(node_map: Mapping[PathT, Node]) -> None:
    unvisited = dict(node_map)
    _check_node_map(path=immutabledict(), unvisited=unvisited)

    if unvisited:
        raise InvalidTreeException("There are orphaned entries in the node map")

def _check_node_map(*, path: immutabledict, unvisited: dict) -> None:
    if path not in unvisited:
        return

    node = unvisited.pop(path)

    if node.label in path.keys():
        raise InvalidTreeException(f"Duplicate label '{label}' found along a path")

    for component_label in node.component_labels:
        path_ = path | {node.label: component_label}
        _check_node_map(path=path_, unvisited=unvisited)
