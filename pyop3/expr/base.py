from __future__ import annotations

import abc
import collections
import functools
import numbers
from functools import cached_property

import numpy as np
from immutabledict import immutabledict as idict

from pyop3 import utils
from pyop3.tree.axis_tree import UNIT_AXIS_TREE, AxisTree, merge_axis_trees


# TODO: define __str__ as an abc?
class Expression(abc.ABC):

    MAX_NUM_CHARS = 120

    # {{{ abstract methods

    # TODO: I reckon that this isn't strictly necessary - we can always detect the shape
    # via a traversal. Similarly for loop axes and such.
    @property
    @abc.abstractmethod
    def shape(self) -> AxisTree:
        pass

    @property
    @abc.abstractmethod
    def loop_axes(self) -> tuple[Axis]:
        pass

    @property
    @abc.abstractmethod
    def _full_str(self) -> str:
        pass

    # }}}

    def __str__(self) -> str:
        full_str = self._full_str
        if len(full_str) > self.MAX_NUM_CHARS:
            pos = self.MAX_NUM_CHARS // 2 - 1
            return f"{full_str[:pos]}..{full_str[-pos:]}"
        else:
            return full_str

    def __add__(self, other):
        if other == 0:
            return self
        else:
            return Add(self, other)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return Add(other, self)

    def __sub__(self, other) -> Sub | Self:
        if other == 0:
            return self
        else:
            return Sub(self, other)

    def __rsub__(self, other) -> Sub | Self:
        if other == 0:
            return self
        else:
            return Sub(other, self)

    def __mul__(self, other) -> Mul | Self:
        if other == 1:
            return self
        else:
            return Mul(self, other)

    def __rmul__(self, other) -> Mul | Self:
        if other == 1:
            return self
        else:
            return Mul(other, self)

    def __floordiv__(self, other) -> FloorDiv | Self:
        if not isinstance(other, numbers.Integral):
            return NotImplemented

        if other == 1:
            return self
        else:
            return FloorDiv(self, other)

    def __mod__(self, other) -> Modulo | Self:
        # TODO: raise nice exception
        assert isinstance(other, numbers.Number)

        if other == 1:
            return self
        else:
            return Modulo(self, other)

    def __neg__(self) -> Neg:
        if isinstance(self, Neg):
            # Neg(Neg(obj)) == obj
            return self.operand
        else:
            return Neg(self)

    def __lt__(self, other):
        from pyop3.expr.visitors import max_value, min_value

        if max_value(self) < min_value(other):
            return True
        elif min_value(self) >= max_value(other):
            return False
        else:
            return LessThan(self, other)

    def __gt__(self, other):
        from pyop3.expr.visitors import max_value, min_value

        if min_value(self) > max_value(other):
            return True
        elif max_value(self) <= min_value(other):
            return False
        else:
            return GreaterThan(self, other)

    def __le__(self, other):
        from pyop3.expr.visitors import max_value, min_value

        if max_value(self) <= min_value(other):
            return True
        elif min_value(self) > max_value(other):
            return False
        else:
            return LessThanOrEqual(self, other)

    def __ge__(self, other):
        from pyop3.expr.visitors import max_value, min_value

        if min_value(self) >= max_value(other):
            return True
        elif max_value(self) < min_value(other):
            return False
        else:
            return GreaterThanOrEqual(self, other)

    def __or__(self, other) -> Or | bool:
        return self._maybe_eager_or(self, other)

    def __ror__(self, other) -> Or | bool:
        return self._maybe_eager_or(other, self)

    @classmethod
    def _maybe_eager_or(cls, a, b) -> Or | Expression | bool:
        from pyop3 import evaluate
        from pyop3.expr.visitors import MissingVariableException  # put in main namespace?

        try:
            a_result = evaluate(a)
        except MissingVariableException:
            a_result = None

        try:
            b_result = evaluate(b)
        except MissingVariableException:
            b_result = None

        if a_result or b_result:
            return True
        elif a_result is False:
            if b_result is False:
                return False
            else:
                assert b_result is None
                return b
        else:
            assert a_result is None
            if b_result is False:
                return a
            else:
                assert b_result is None
                return Or(a, b)

    @cached_property
    def max_value(self) -> numbers.Number:
        from pyop3.expr.visitors import max_value

        return max_value(self)

    @cached_property
    def min_value(self) -> numbers.Number:
        from pyop3.expr.visitors import min_value

        return min_value(self)


class Operator(Expression, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def operands(self):
        pass



class UnaryOperator(Operator, metaclass=abc.ABCMeta):

    # {{{ interface impls

    @property
    def _full_str(self) -> str:
        return f"{self.symbol}{as_str(self.a)}"

    # }}}

    @property
    def operands(self):
        return (self.a,)

    @property
    def operand(self):
        return utils.just_one(self.operands)

    @property
    @abc.abstractmethod
    def symbol(self) -> str:
        pass

    def __init__(self, a, /) -> None:
        self.a = a

    @property
    def shape(self):
        return self.a.shape

    @property
    def loop_axes(self):
        return self.a.loop_axes


class Neg(UnaryOperator):
    @property
    def symbol(self) -> str:
        return "-"


class BinaryOperator(Operator, metaclass=abc.ABCMeta):
    @property
    def operands(self):
        return (self.a, self.b)

    def __init__(self, a, b, /):
        self.a = a
        self.b = b

    @cached_property
    def shape(self) -> tuple[AxisTree]:
        from pyop3.expr.visitors import get_shape

        return (
            merge_axis_trees((
                utils.just_one(get_shape(self.a)),
                utils.just_one(get_shape(self.b)),
            )),
        )

    @cached_property
    def loop_axes(self):
        from pyop3.expr.visitors import get_loop_axes

        a_loop_axes = get_loop_axes(self.a)
        b_loop_axes = get_loop_axes(self.b)
        axes = collections.defaultdict(tuple, a_loop_axes)
        for loop_index, loop_axes in b_loop_axes.items():
            axes[loop_index] = utils.unique((*axes[loop_index], *loop_axes))
        return idict(axes)

    def __hash__(self) -> int:
        return hash((type(self), self.a, self.b))

    def __eq__(self, other, /) -> bool:
        return type(self) == type(other) and other.a == self.a  and other.b == self.b

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.a!r}, {self.b!r})"

    @property
    def _full_str(self) -> str:
        # Always use brackets to avoid having to deal with operator precedence rules
        return f"({as_str(self.a)} {self._symbol} {as_str(self.b)})"

    @property
    @abc.abstractmethod
    def _symbol(self) -> str:
        pass

    # def as_str(self, *, full=True) -> str:
    #     # Always use brackets to avoid having to deal with operator precedence rules
    #     return f"({self.as_str(a, full=full)} {self._symbol} {self.as_str(b, full=full)})"


class Add(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "+"

    # def __init__(self, *args):
    #     super().__init__(*args)
    #
    #     if "array_22" in str(self):
    #         breakpoint()


class Sub(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "-"


class Mul(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "*"


class FloorDiv(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "//"


class Modulo(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "%"


class Condition(Operator, metaclass=abc.ABCMeta):
    pass


class BinaryCondition(BinaryOperator, Condition, metaclass=abc.ABCMeta):
    pass


class LessThan(BinaryCondition):
    @property
    def _symbol(self) -> str:
        return "<"


class GreaterThan(BinaryCondition):
    @property
    def _symbol(self) -> str:
        return ">"


class LessThanOrEqual(BinaryCondition):
    @property
    def _symbol(self) -> str:
        return "<="


class GreaterThanOrEqual(BinaryCondition):
    @property
    def _symbol(self) -> str:
        return ">="


class Or(BinaryCondition):

    @property
    def _symbol(self) -> str:
        return "|"


class TernaryOperator(Operator, metaclass=abc.ABCMeta):
    @property
    def operands(self):
        return (self.a, self.b, self.c)

    def __init__(self, a, b, c, /) -> None:
        self.a = a
        self.b = b
        self.c = c

    def __eq__(self, other):
        return type(other) == type(self) and other.operands == self.operands

    def __hash__(self):
        return hash((type(self), self.operands))


class Conditional(TernaryOperator):

    # {{{ interface impls

    # TODO: should have a way of truncating here to keep a bit of all of them, maybe we also need a "short str" method?
    def __str__(self) -> str:
        return f"{self.predicate} ? {self.if_true} : {self.if_false}"

    @property
    def _full_str(self) -> str:
        return f"{as_str(self.predicate)} ? {as_str(self.if_true)} : {as_str(self.if_false)}"

    # }}}

    @property
    def shape(self):
        from pyop3.expr.visitors import get_shape

        trees = (utils.just_one(get_shape(o)) for o in self.operands)
        return (merge_axis_trees(trees),)

        # if not isinstance(self.if_true, numbers.Number):
        #     true_shape = get_shape(self.if_true)
        #     if not isinstance(self.if_false, numbers.Number):
        #         false_shape = self.if_false.shape
        #         return utils.single_valued((true_shape, false_shape))
        #     else:
        #         return true_shape
        # else:
        #     return get_shape(self.if_false)

    @cached_property
    def loop_axes(self) -> tuple[Axis]:
        from pyop3.expr.visitors import get_loop_axes

        a_loop_axes = get_loop_axes(self.a)
        b_loop_axes = get_loop_axes(self.b)
        c_loop_axes = get_loop_axes(self.b)
        axes = collections.defaultdict(tuple, a_loop_axes)
        for loop_index, loop_axes in b_loop_axes.items():
            axes[loop_index] = utils.unique((*axes[loop_index], *loop_axes))
        for loop_index, loop_axes in c_loop_axes.items():
            axes[loop_index] = utils.unique((*axes[loop_index], *loop_axes))
        return idict(axes)

    @property
    def predicate(self):
        return self.a

    @property
    def if_true(self):
        return self.b

    @property
    def if_false(self):
        return self.c


def conditional(predicate, if_true, if_false):
    from pyop3 import evaluate
    from pyop3.expr.visitors import MissingVariableException  # put in main namespace?

    # Try to simplify by eagerly evaluating the operands

    # If both branches are the same then just return one of them.
    if if_true == if_false:
        return if_true

    # Attempt to eagerly evaluate 'predicate' to avoid creating
    # unnecessary objects.
    try:
        return if_true if evaluate(predicate) else if_false
    except MissingVariableException:
        return Conditional(predicate, if_true, if_false)


class Terminal(Expression, abc.ABC):

    def __hash__(self) -> int:
        return hash((type(self), self.terminal_key))

    def __eq__(self, other, /) -> bool:
        return type(self) == type(other) and other.terminal_key == self.terminal_key

    @property
    @abc.abstractmethod
    def terminal_key(self):
        # used in `replace_terminals()`
        pass


class AxisVar(Terminal):

    # {{{ interface impls

    @cached_property
    def shape(self) -> tuple[AxisTree]:
        from pyop3.tree.axis_tree.tree import full_shape
        return (merge_axis_trees((full_shape(self.axis.as_tree())[0], self.axis.as_tree())),)

    loop_axes = idict()

    @property
    def _full_str(self) -> str:
        return f"i_{{{self.axis_label}}}"

    # }}}

    def __init__(self, axis: Axis) -> None:
        assert len(axis.components) == 1
        assert axis.component.sf is None
        assert tuple(r.label for r in axis.component.regions) == (None,)
        self.axis = axis

    # TODO: when we use frozenrecord
    # def __post_init__(self) -> None:
    #     assert self.axis.is_linear

    # TODO: deprecate?
    @property
    def axis_label(self):
        return self.axis.label

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.axis_label!r})"

    @property
    def terminal_key(self) -> str:
        return self.axis_label


class NaN(Terminal):
    # {{{ interface impls

    shape = UNIT_AXIS_TREE
    loop_axes = ()

    _full_str = "NaN"

    # }}}

    def __repr__(self) -> str:
        return "NaN()"

    @property
    def terminal_key(self) -> str:
        return str(self)


NAN = NaN()


# TODO: Refactor so loop ID passed in not the actual index
class LoopIndexVar(Terminal):

    # {{{ interface impls

    @property
    def shape(self):
        return (UNIT_AXIS_TREE,)

    @property
    def loop_axes(self):
        return idict({self.loop_index: (self.axis,)})

    @property
    def _full_str(self) -> str:
        return f"L_{{{self.loop_index.id}, {self.axis_label}}}"

    # }}}

    def __init__(self, loop_index, axis) -> None:
        from pyop3 import LoopIndex

        assert not isinstance(axis, str), "changed"
        assert isinstance(loop_index, LoopIndex)
        assert axis.component.sf is None
        self.loop_index = loop_index
        self.axis = axis

        # we must be linear at this point
        assert len(self.axis.components) == 1

    # TODO: deprecate me
    @property
    def axis_label(self):
        return self.axis.label

    @property
    def loop_id(self):
        return self.loop_index.id

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.loop_index!r}, {self.axis_label!r})"

    @property
    def terminal_key(self) -> tuple:
        return (self.loop_index.id, self.axis_label)


ExpressionT = Expression | numbers.Number


@functools.singledispatch
def as_str(expr):
    return expr._full_str


@as_str.register(Expression)
def _(expr):
    return expr._full_str


@as_str.register(numbers.Number)
@as_str.register(bool)
@as_str.register(np.bool)
def _(expr):
    return str(expr)


def get_loop_tree(expr) -> tuple[AxisTree, Mapping[LoopIndexVar, AxisVar]]:
    from pyop3.expr.visitors import get_loop_axes

    axes = []
    loop_var_replace_map = {}
    for loop_index, loop_axes in get_loop_axes(expr).items():
        for loop_axis in loop_axes:
            axis_label = f"{loop_axis.label}_{loop_index.id}"

            axis = loop_axis.copy(label=axis_label)
            axes.append(axis)

            loop_var = LoopIndexVar(loop_index, loop_axis)
            axis_var = AxisVar(axis)
            loop_var_replace_map[loop_var] = axis_var

    return (AxisTree.from_iterable(axes), loop_var_replace_map)


def loopified_shape(expr: Expression) -> tuple[AxisTree, Mapping[LoopIndexVar, AxisVar]]:
    from pyop3.expr.visitors import replace, get_shape

    loop_tree, loop_var_replace_map = get_loop_tree(expr)

    # assume single tree for now
    shape = utils.just_one(get_shape(expr))

    if shape is UNIT_AXIS_TREE:
        axis_tree = loop_tree
    else:
        # Replace any references to the loop indices
        new_node_map = {}
        for path, axis in shape.node_map.items():
            if axis is None:
                new_node_map[path] = None
                continue

            new_components = []
            for component in axis.components:
                new_regions = []
                for region in component.regions:
                    new_size = replace(region.size, loop_var_replace_map)
                    new_regions.append(region.__record_init__(size=new_size))
                new_components.append(component.copy(regions=new_regions))
            new_node_map[path] = axis.copy(components=new_components)
        subtree = AxisTree(new_node_map)
        axis_tree = loop_tree.add_subtree(loop_tree.leaf_path, subtree)

    assert not axis_tree._all_region_labels

    return axis_tree, loop_var_replace_map
