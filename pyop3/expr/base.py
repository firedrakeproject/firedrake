from __future__ import annotations

import abc
import collections
import functools
import numbers
from functools import cached_property
from typing import NoReturn

import numpy as np
from immutabledict import immutabledict as idict
from mpi4py import MPI

import pyop3.collections
import pyop3.record
from pyop3 import utils
from pyop3.node import Node, Terminal
from pyop3.axis_tree import UNIT_AXIS_TREE, AxisTree, merge_axis_trees
from pyop3.axis_tree.tree import MissingVariableException


class Expression(Node, abc.ABC):

    # {{{ abstract methods

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _full_str(self) -> str:
        pass

    # }}}

    def __str__(self) -> str:
        return self._full_str

    # {{{ arithmetic

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

    def __truediv__(self, other) -> Div | Self:
        if other == 1:
            return self
        else:
            return Div(self, other)

    def __rtruediv__(self, other) -> Div | Self:
        return Div(other, self)

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

    def __pow__(self, other, /) -> Pow:
        return Pow(self, other)

    def __abs__(self) -> Abs:
        if isinstance(self, Abs):
            # Abs(Abs(obj)) == Abs(obj)
            return self.operand
        else:
            return Abs(self)

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

    # }}}

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


class Operator(Expression, metaclass=abc.ABCMeta):

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def operands(self) -> tuple[ExpressionT, ...]:
        pass

    # }}}


@pyop3.record.frozenrecord()
class UnaryOperator(Operator, metaclass=abc.ABCMeta):

    # {{{ instance attrs

    a: ExpressionT

    def collect_buffers(self, visitor):
        return visitor(self.a)

    def get_disk_cache_key(self, visitor):
        return (type(self), visitor(self.a))

    get_instruction_executor_cache_key = get_disk_cache_key

    # }}}

    # {{{ interface impls

    @property
    def operands(self) -> tuple[ExpressionT]:
        return (self.a,)

    child_attrs = ("a",)

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

    @property
    def local_max(self) -> numbers.Number:
        return -self.a.local_min

    @property
    def local_min(self) -> numbers.Number:
        return -self.a.local_max


class Abs(UnaryOperator):
    @property
    def symbol(self) -> str:
        return "abs."

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError


@pyop3.record.frozenrecord()
class BinaryOperator(Operator, metaclass=abc.ABCMeta):

    # {{{ instance attrs

    a: ExpressionT
    b: ExpressionT

    def collect_buffers(self, visitor):
        return visitor(self.a) | visitor(self.b)

    def get_disk_cache_key(self, visitor):
        return (type(self), visitor(self.a), visitor(self.b))

    get_instruction_executor_cache_key = get_disk_cache_key

    # }}}

    # {{{ interface impls

    child_attrs = ("a", "b")

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

    # {{{ interface impls

    @property
    def _symbol(self) -> str:
        return "+"

    @property
    def local_max(self) -> numbers.Number:
        from pyop3.expr.visitors import get_local_max

        return get_local_max(self.a) + get_local_max(self.b)

    @property
    def local_min(self) -> numbers.Number:
        from pyop3.expr.visitors import get_local_min

        return get_local_min(self.a) + get_local_min(self.b)

    # }}}


class Sub(BinaryOperator):

    # {{{ interface impls

    @property
    def _symbol(self) -> str:
        return "-"

    @property
    def local_max(self) -> numbers.Number:
        from pyop3.expr.visitors import get_local_max, get_local_min

        return get_local_max(self.a) - get_local_min(self.b)

    @property
    def local_min(self) -> numbers.Number:
        from pyop3.expr.visitors import get_local_max, get_local_min

        return get_local_min(self.a) - get_local_max(self.b)

    # }}}


class Mul(BinaryOperator):

    # {{{ interface impls

    @property
    def _symbol(self) -> str:
        return "*"

    @property
    def local_max(self) -> numbers.Number:
        from pyop3.expr.visitors import get_local_max

        return get_local_max(self.a) * get_local_max(self.b)

    @property
    def local_min(self) -> numbers.Number:
        from pyop3.expr.visitors import get_local_min

        return get_local_min(self.a) * get_local_min(self.b)

    # }}}


class Div(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "/"


class FloorDiv(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "//"


class Modulo(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "%"


class Pow(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "**"

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError


class Comparison(BinaryOperator, metaclass=abc.ABCMeta):

    # {{{ interface impls

    @property
    def local_max(self) -> numbers.Number:
        raise TypeError("not sure that this makes sense")

    @property
    def local_min(self) -> numbers.Number:
        raise TypeError("not sure that this makes sense")

    # }}}


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


@pyop3.record.frozenrecord()
class TernaryOperator(Operator, metaclass=abc.ABCMeta):

    # {{{ instance attrs

    a: ExpressionT
    b: ExpressionT
    c: ExpressionT

    def collect_buffers(self, visitor):
        return visitor(self.a) | visitor(self.b) | visitor(self.c)

    def get_disk_cache_key(self, visitor):
        return (type(self), visitor(self.a), visitor(self.b), visitor(self.c))

    get_instruction_executor_cache_key = get_disk_cache_key

    # }}}

    # {{{ interface impls

    child_attrs = ("a", "b", "c")

    @property
    def operands(self) -> tuple[ExpressionT, ExpressionT, ExpressionT]:
        return (self.a, self.b, self.c)

    # }}}


@pyop3.record.frozenrecord()
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

    @property
    def local_max(self) -> numbers.Number:
        raise TypeError("not sure that this makes sense")
        from pyop3.expr.visitors import get_local_max

        return max(*map(get_local_max, [self.if_true, self.if_false]))

    @property
    def local_min(self) -> numbers.Number:
        raise TypeError("not sure that this makes sense")



def conditional(predicate, if_true, if_false):
    from pyop3 import evaluate

    if if_true == if_false:
        return if_true

    try:
        predicate = evaluate(predicate)
    except MissingVariableException:
        return Conditional(predicate, if_true, if_false)
    else:
        assert isinstance(predicate, bool)
        return if_true if predicate else if_false


class TerminalExpression(Expression, Terminal, abc.ABC):

    child_attrs = ()


class NamedTerminalExpression(TerminalExpression):
    """A terminal with a name.

    This type is important because only named terminals can be replaced when
    an operation is reused. For example we can only do the following:

        loop = op3.loop(p, kernel(dat1[p]))
        loop(**{"dat": dat2})  # pass dat2 instead of dat1

    if ``dat1`` is a named terminal.

    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass


@pyop3.record.frozenrecord()
class AxisVar(TerminalExpression):

    # {{{ instance attrs

    axis: Axis

    def collect_buffers(self, visitor):
        # Axis vars are just pointers to some outer loop. Any internal
        # buffers that we need will be referenced elsewhere.
        return pyop3.collections.OrderedFrozenSet()

    def get_disk_cache_key(self, visitor) -> Hashable:
        # Axis vars are just pointers to some outer loop. We don't
        # need to recurse here, just make sure that the labels match.
        return (
            type(self),
            ("axis", visitor.renamer.add(self.axis.label, "Axis")),
        )

    get_instruction_executor_cache_key = get_disk_cache_key

    @classmethod
    def record_prepare_args(cls, axis: Axis) -> None:
        assert len(axis.components) == 1
        assert axis.component.sf is None
        # FIXME
        # assert tuple(r.label for r in axis.component.regions) == (None,)

        return dict(axis=axis)

    # }}}

    # {{{ interface impls

    @property
    def local_max(self) -> numbers.Number:
        raise TypeError("not sure that this makes sense")

    @property
    def local_min(self) -> numbers.Number:
        raise TypeError("not sure that this makes sense")

    @property
    def _full_str(self) -> str:
        return f"i_{{{self.axis.label}}}"

    # }}}


@pyop3.record.frozenrecord()
class NaN(TerminalExpression):

    # {{{ pyop3.obj.Object interface impls

    def disk_cache_key(self, renamer):
        return (type(self),)

    instruction_executor_cache_key = disk_cache_key

    @classmethod
    def get_custom_comm(cls) -> MPI.Comm:
        return MPI.COMM_SELF

    # }}}

    # {{{ interface impls


    @property
    def local_max(self) -> NoReturn:
        raise TypeError

    @property
    def local_min(self) -> NoReturn:
        raise TypeError

    _full_str = "NaN"

    # }}}


NAN = NaN()


@pyop3.record.frozenrecord()
class LoopIndexVar(TerminalExpression):

    # {{{ instance attrs

    loop_index: LoopIndex
    axis: Axis

    def collect_buffers(self, visitor):
        # Loop index vars are just pointers to some outer loop. Any internal
        # buffers that we need will be referenced elsewhere.
        return pyop3.collections.OrderedFrozenSet()

    def get_disk_cache_key(self, visitor) -> Hashable:
        # Loop index vars are just pointers to some outer loop. We don't
        # need to recurse here, just make sure that the labels match.
        return (
            type(self),
            visitor.renamer.add(self.loop_index.id, "LoopIndex"),
            visitor.renamer.add(self.axis.label, "Axis"),
        )

    get_instruction_executor_cache_key = get_disk_cache_key

    @classmethod
    def record_prepare_args(cls, loop_index, axis) -> None:
        from pyop3 import LoopIndex

        # we must be linear at this point
        assert len(axis.components) == 1

        assert isinstance(loop_index, LoopIndex)
        assert axis.component.sf is None
        return dict(loop_index=loop_index, axis=axis)

    # }}}

    # {{{ interface impls

    @property
    def local_max(self) -> numbers.Number:
        raise TypeError("not sure that this makes sense")

    @property
    def local_min(self) -> numbers.Number:
        raise TypeError("not sure that this makes sense")

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
        new_axis = axis.record_new(_label=new_axis_label)
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
                    new_regions.append(region.record_new(size=new_size))
                new_regions = tuple(new_regions)
                new_components.append(component.record_new(regions=new_regions))
            new_node_map[path] = axis.record_new(components=tuple(new_components))
        subtree = AxisTree(new_node_map)
        axis_tree = loop_tree.add_subtree(loop_tree.leaf_path, subtree)

    assert not axis_tree._all_region_labels

    return axis_tree, loop_var_replace_map
