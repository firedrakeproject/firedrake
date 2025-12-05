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

    def __add__(self, other: ExpressionT, /) -> Expression:
        if other == 0:
            return self
        else:
            return Add(self, other)

    def __radd__(self, other: ExpressionT, /) -> Expression:
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
        return LessThan(self, other)

    def __gt__(self, other):
        return GreaterThan(self, other)

    def __le__(self, other):
        return LessThanOrEqual(self, other)

    def __ge__(self, other):
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
        from pyop3.expr.visitors import find_max_value

        return find_max_value(self)

    @cached_property
    def min_value(self) -> numbers.Number:
        from pyop3.expr.visitors import find_min_value

        return find_min_value(self)


class Operator(Expression, metaclass=abc.ABCMeta):

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def operands(self) -> tuple[ExpressionT, ...]:
        pass

    # }}}


@utils.frozenrecord()
class UnaryOperator(Operator, metaclass=abc.ABCMeta):

    # {{{ instance attrs

    a: ExpressionT

    # }}}

    # {{{ interface impls

    @property
    def operands(self) -> tuple[ExpressionT]:
        return (self.a,)

    @property
    def _full_str(self) -> str:
        return f"{self.symbol}{as_str(self.a)}"

    # }}}

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def symbol(self) -> str:
        pass

    # }}}

    @property
    def operand(self):
        return utils.just_one(self.operands)


class Neg(UnaryOperator):
    @property
    def symbol(self) -> str:
        return "-"


@utils.frozenrecord()
class BinaryOperator(Operator, metaclass=abc.ABCMeta):

    # {{{ instance attrs

    a: ExpressionT
    b: ExpressionT

    # }}}

    # {{{ interface impls

    @property
    def operands(self) -> tuple[ExpressionT, ExpressionT]:
        return (self.a, self.b)

    # }}}

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def _symbol(self) -> str:
        pass


    # }}}

    @property
    def _full_str(self) -> str:
        # Always use brackets to avoid having to deal with operator precedence rules
        return f"({as_str(self.a)} {self._symbol} {as_str(self.b)})"


class Add(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "+"


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


class Comparison(BinaryOperator, metaclass=abc.ABCMeta):
    pass


class LessThan(Comparison):
    @property
    def _symbol(self) -> str:
        return "<"


class GreaterThan(Comparison):
    @property
    def _symbol(self) -> str:
        return ">"


class LessThanOrEqual(Comparison):
    @property
    def _symbol(self) -> str:
        return "<="


class GreaterThanOrEqual(Comparison):
    @property
    def _symbol(self) -> str:
        return ">="


class Or(Comparison):

    @property
    def _symbol(self) -> str:
        return "|"


@utils.frozenrecord()
class TernaryOperator(Operator, metaclass=abc.ABCMeta):

    # {{{ instance attrs

    a: ExpressionT
    b: ExpressionT
    c: ExpressionT

    # }}}

    # {{{ interface impls

    @property
    def operands(self) -> tuple[ExpressionT, ExpressionT, ExpressionT]:
        return (self.a, self.b, self.c)

    # }}}


@utils.frozenrecord()
class Conditional(TernaryOperator):

    # {{{ interface impls

    @property
    def _full_str(self) -> str:
        return f"{as_str(self.predicate)} ? {as_str(self.if_true)} : {as_str(self.if_false)}"

    # }}}

    @property
    def predicate(self) -> ExpressionT:
        return self.a

    @property
    def if_true(self) -> ExpressionT:
        return self.b

    @property
    def if_false(self) -> ExpressionT:
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
    pass


@utils.frozenrecord()
class AxisVar(Terminal):

    axis: Axis

    # {{{ interface impls

    @property
    def _full_str(self) -> str:
        return f"i_{{{self.axis_label}}}"

    # }}}

    def __init__(self, axis: Axis) -> None:
        assert len(axis.components) == 1
        assert axis.component.sf is None
        assert tuple(r.label for r in axis.component.regions) == (None,)
        object.__setattr__(self, "axis", axis)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.axis.label!r})"


@utils.frozenrecord()
class NaN(Terminal):
    # {{{ interface impls

    _full_str = "NaN"

    # }}}


NAN = NaN()


@utils.frozenrecord()
class LoopIndexVar(Terminal):

    loop_index: LoopIndex
    axis: Axis

    def __init__(self, loop_index, axis) -> None:
        from pyop3 import LoopIndex

        # we must be linear at this point
        assert len(axis.components) == 1

        assert isinstance(loop_index, LoopIndex)
        assert axis.component.sf is None
        object.__setattr__(self, "loop_index", loop_index)
        object.__setattr__(self, "axis", axis)

    # {{{ interface impls

    @property
    def _full_str(self) -> str:
        return f"L_{{{self.loop_index.id}, {self.axis.label}}}"

    # }}}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.loop_index!r}, {self.axis.label!r})"


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
    from pyop3.expr.visitors import collect_loop_index_vars

    axes = []
    loop_var_replace_map = {}
    for loop_var in collect_loop_index_vars(expr):
        axis = loop_var.axis
        new_axis_label = f"{axis.label}_{loop_var.loop_index.id}"
        new_axis = axis.copy(label=new_axis_label)
        axes.append(new_axis)
        loop_var_replace_map[loop_var] = AxisVar(new_axis)
    return (AxisTree.from_iterable(axes), loop_var_replace_map)


def loopified_shape(expr: Expression) -> tuple[AxisTree, Mapping[LoopIndexVar, AxisVar]]:
    from pyop3.expr.visitors import replace, get_shape

    loop_tree, loop_var_replace_map = get_loop_tree(expr)

    # assume single tree for now
    shape = utils.just_one(get_shape(expr))

    if shape is UNIT_AXIS_TREE:
        if loop_tree:
            axis_tree = loop_tree
        else:
            axis_tree = UNIT_AXIS_TREE
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
