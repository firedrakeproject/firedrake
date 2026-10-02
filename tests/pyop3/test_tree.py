import pytest

from pyop3.tree import *

# This file is pretty outdated
pytest.skip(allow_module_level=True)


def test_parent_works_with_nodes_and_ids():
    a = Node("a")
    b = Node("b")
    tree = Tree(a)
    tree.add_node(b, parent=a)

    assert tree.parent(b) == a
    assert tree.parent("b") == a


def test_children_works_with_nodes_and_ids():
    a = Node("a")
    b = Node("b")
    tree = Tree(a)
    tree.add_node(b, parent=a)

    assert tree.children(a) == (b,)
    assert tree.children("a") == (b,)


def test_tree_root_has_no_parent():
    a = Node("a")
    tree = Tree(a)
    assert tree.parent(a) is None


def test_tree_is_empty():
    tree = Tree()
    assert tree.is_empty
    tree.add_node(Node("a"))
    assert not tree.is_empty


@pytest.mark.skip("not sure if we still want this")
def test_can_set_root_multiple_times():
    tree = Tree()
    tree.root = Node("a")
    assert tree.root.id == "a"
    tree.root = Node("b")
    assert tree.root.id == "b"


def test_cannot_add_another_root():
    tree = Tree(Node("a"))
    with pytest.raises(ValueError):
        tree.add_node(Node("b"))


def test_add_node():
    a = Node("a")
    b = Node("b")
    tree = Tree(a)
    tree.add_node(b, parent=a)

    assert tree.children(a) == (b,)
    assert tree.parent(b) == a
    assert tree.children(b) == ()


@pytest.mark.parametrize("bulk", [True, False])
def test_add_multiple_children(bulk):
    a = Node("a")
    b = Node("b")
    c = Node("c")

    tree = Tree(a)
    if bulk:
        tree.add_node(b, parent=a)
        tree.add_node(c, parent=a)
    else:
        tree.add_nodes([b, c], parent=a)

    assert tree.children(a) == (b, c)
    assert tree.parent(b) == a
    assert tree.parent(c) == a
    assert tree.children(b) == ()
    assert tree.children(c) == ()


@pytest.fixture
def treeA():
    a = Node("a")
    b = Node("b")
    c = Node("c")
    d = Node("d")
    e = Node("e")
    f = Node("f")

    tree = Tree(a)
    tree.add_nodes([b, c], parent=a)
    tree.add_nodes([d, e], parent=b)
    tree.add_node(f, parent=c)
    return tree


@pytest.fixture
def tree2():
    tree = Tree()
    x = Node("x")
    y = Node("y")
    z = Node("z")
    tree.add_node(x)
    tree.add_nodes([y, z], x)
    return tree


@pytest.fixture
def tree3():
    tree = Tree()
    one = Node(1)
    two = Node(2)
    tree.add_node(one)
    tree.add_node(two, one)
    return tree


def test_tree_str(treeA):
    assert (
        str(treeA)
        == """\
Node(id='a')
├──➤ Node(id='b')
│    ├──➤ Node(id='d')
│    └──➤ Node(id='e')
└──➤ Node(id='c')
     └──➤ Node(id='f')"""
    )


def test_tree_depth():
    tree = Tree()
    assert tree.depth == 0
    tree.add_node(Node("a"))
    assert tree.depth == 1
    tree.add_node(Node("b"), "a")
    assert tree.depth == 2
    tree.add_node(Node("c"), "a")
    assert tree.depth == 2


def test_tree_copy(treeA):
    treeB = treeA.copy()
    assert treeA.depth == treeB.depth == 3
    assert str(treeA) == str(treeB)

    treeA.add_node(Node("g"), "e")
    assert treeA.depth == 4
    assert treeB.depth == 3


def test_pop_subtree(treeA):
    # Test that popping 'b' from the tree
    #
    #     Node(id='a')
    #     ├──➤ Node(id='b')
    #     │    ├──➤ Node(id='d')
    #     │    └──➤ Node(id='e')
    #     └──➤ Node(id='c')
    #          └──➤ Node(id='f')
    #
    # returns the subtree
    #
    #     Node(id='b')
    #     ├──➤ Node(id='d')
    #     └──➤ Node(id='e')
    #
    # and changes the original tree to
    #
    #     Node(id='a')
    #     └──➤ Node(id='c')
    #          └──➤ Node(id='f')

    subtree = treeA.pop_subtree("b")
    assert subtree.depth == 2
    assert subtree.root.id == "b"
    assert subtree.children("b") == (subtree.find("d"), subtree.find("e"))
    assert not subtree.children("d") and not subtree.children("e")

    assert treeA.depth == 3
    assert treeA.root.id == "a"
    assert treeA.children("a") == (treeA.find("c"),)
    assert treeA.children("c") == (treeA.find("f"),)
    assert not treeA.children("f")


def test_add_subtree():
    a = Node("a")
    b = Node("b")
    c = Node("c")

    tree = Tree(a)
    tree.add_nodes([b, c], a)
    assert tree.depth == 2

    x = Node("x")
    y = Node("y")
    subtree = Tree(x)
    subtree.add_node(y, x)
    assert subtree.depth == 2

    tree.add_subtree(subtree, "b")

    assert tree.depth == 4
    assert tree.children("a") == (b, c)
    assert tree.children("b") == (x,)
    assert tree.children("c") == ()
    assert tree.children("x") == (y,)
    assert tree.children("y") == ()


def test_add_subtree_with_uniquified_matching_ids():
    a = Node("a")
    b = Node("b")
    tree = Tree(a)
    tree.add_node(b, a)
    subtree = Tree(b)

    tree.add_subtree(subtree, a, uniquify=True)

    assert tree.depth == 2
    child1, child2 = tree.children(a)
    assert child1 is not child2
    assert child1.id == "b"
    assert child2.id == "b_0"


def test_add_subtree_with_matching_ids_fails_without_uniquify():
    a = Node("a")
    b = Node("b")
    tree = Tree(a)
    tree.add_node(b, a)
    subtree = Tree(b)

    with pytest.raises(ValueError):
        tree.add_subtree(subtree, a, uniquify=False)


@pytest.mark.skip("Not sure on the right API")
def test_tree_construction_from_nested_list():
    # Create a tree corresponding to:
    #
    #     Node(id='a')
    #     ├──➤ Node(id='b')
    #     │    ├──➤ Node(id='d')
    #     │    └──➤ Node(id='e')
    #     └──➤ Node(id='c')
    #          └──➤ Node(id='f')
    nodes = [
        RangeNode("a"),
        [
            [RangeNode("b"), [RangeNode("c"), RangeNode("d")]],
            [RangeNode("e"), [RangeNode("f")]],
        ],
    ]
    nodes = {
        Node("a"): ("b", "c"),
        Node("b"): ("d", "e"),
        Node("c"): ("f",),
        Node("d"): (),
        Node("e"): (),
        Node("f"): (),
    }
    tree = Tree(nodes)

    assert False
