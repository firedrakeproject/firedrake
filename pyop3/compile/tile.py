"""Lower GEM instructions to a cuTile kernel.

`pyop3.compile.mlir` translates a scheduled Impero tree into scalar code, with one
thread per item. This module translates the GEM instructions directly: each block
of the kernel handles a batch of items, and each GEM value is a tile that has one
axis per free index.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import linecache
import numbers
from functools import cached_property
from typing import Any

import cupy as cp
import numpy as np

from gem import gem, impero, impero_utils, interpreter
from gem.node import pre_traversal

from pyop3 import utils
from pyop3.buffer import NullBuffer
from pyop3.compile.gem import GemExecutable
from pyop3.compile.passes.impero.item_loop import needs_atomic
from pyop3.compile.passes.impero.utils import is_temporary, static_extent
from pyop3.device import HOST_DEVICE, get_current_device, offloading

MAX_ITEMS_PER_BLOCK = 128
"""The largest number of items that a block handles. Tile shapes must be powers of two."""

# shortcut: measured with tileiras 13.4 on sm_89 only, where kernels with a tile of 2**16 entries
# can fail to compile; measure again for another compiler or device.
MAX_TILE_SIZE = 2**15
"""The largest number of entries of a tile"""


def ct(name: str, *args: ast.expr, **kwargs: ast.expr) -> ast.expr:
    """The expression `ct.name`, or the call `ct.name(*args, **kwargs)` if it has arguments"""
    function = ast.Attribute(ast.Name("ct"), name)
    if not args and not kwargs:
        return function
    return ast.Call(function, list(args), [ast.keyword(k, v) for k, v in kwargs.items()])


def padded(extent: int) -> int:
    """The smallest power of two that is not less than `extent`"""
    return 1 << (extent - 1).bit_length()


def nested(array: np.ndarray) -> tuple | np.generic:
    """The entries of `array` as nested tuples of NumPy scalars"""
    return tuple(map(nested, array)) if array.ndim else array[()]


def variables(*exprs: gem.Node) -> dict[str, gem.Variable]:
    """Map the name of each variable below `exprs` to the variable"""
    return {n.name: n for n in pre_traversal(exprs) if isinstance(n, gem.Variable)}


@dataclasses.dataclass(frozen=True)
class Tile:
    """An expression of the kernel with the free index of each of its dimensions.

    The axes follow the order of `TileBuilder._order`, so two tiles are broadcast
    against each other by a reshape only.
    """

    expr: ast.expr
    axes: tuple[gem.Index, ...] = ()


class TileBuilder:
    """Translate GEM assignments over the items of one parallel loop into a cuTile kernel"""

    def __init__(self, instructions, item: gem.Index, items_per_block: int, name: str = "tile_kernel") -> None:
        """
        :arg instructions: The `gem.impero.Assignment`s, which run in order for each item.
        :arg item: The index of the parallel loop.
        :arg items_per_block: The number of items that each block handles, a power of two.
        :arg name: The name of the generated kernel.
        """
        self._instructions = instructions
        self._item = item
        self._items_per_block = items_per_block
        self._name = name

        self._body: list[ast.stmt] = []
        # Name -> value of each global of the kernel
        self.constants: dict[str, Any] = {}
        self._names = utils.UniqueNameGenerator()

        # Free index -> position in the axis order of every tile
        self._order: dict[gem.Index, int] = {item: 0}
        # Free index -> tile of its values. The items of a block are the first axis.
        first = ast.BinOp(ct("bid", ast.Constant(0)), ast.Mult(), ast.Constant(items_per_block))
        self._axes: dict[gem.Index, Tile] = {
            item: self.bind("item", ast.BinOp(first, ast.Add(), self.arange(items_per_block)), (item,))
        }
        # GEM node -> tile, for the instruction that is being translated
        self._values: dict[gem.Node, Tile] = {}

    def build(self, parameters: list[str]) -> str:
        """Return the source of the kernel, which takes the number of items and then `parameters`"""
        for instruction in self._instructions:
            self._values = {}
            self.assign(instruction)

        arguments = [ast.arg("num_items", ct("ScalarInt64")), *map(ast.arg, parameters)]
        function = ast.FunctionDef(self._name, ast.arguments(args=arguments), self._body, [])
        return ast.unparse(ast.fix_missing_locations(ast.Module([function])))

    # {{{ helpers

    def bind(self, prefix: str, expr: ast.expr, axes: tuple[gem.Index, ...] = ()) -> Tile:
        """Assign `expr` to a new variable of the kernel"""
        name = self._names(prefix)
        self._body.append(ast.Assign([ast.Name(name)], expr))
        return Tile(ast.Name(name), axes)

    def constant(self, value: Any) -> ast.expr:
        """Make `value` a global of the kernel.

        cuTile gives a float in the source of a kernel the type float32, and gives
        a NumPy scalar the type of its dtype.
        """
        name = self._names("c")
        self.constants[name] = value
        return ast.Name(name)

    def arange(self, size: int) -> ast.expr:
        return ct("arange", ast.Constant(size), dtype=ct("int64"))

    def size(self, index: gem.Index) -> int:
        """The length of the axis of `index`"""
        return self._items_per_block if index == self._item else padded(static_extent(index))

    def union(self, *tiles: Tile) -> tuple[gem.Index, ...]:
        """The axes of the tile that `tiles` broadcast to"""
        return tuple(sorted({a for t in tiles for a in t.axes}, key=self._order.__getitem__))

    def expand(self, tile: Tile, axes: tuple[gem.Index, ...]) -> ast.expr:
        """Return `tile` with a dimension of length one for each of `axes` that it does not have"""
        if not tile.axes or tile.axes == axes:
            return tile.expr
        shape = [self.size(a) if a in tile.axes else 1 for a in axes]
        return ct("reshape", tile.expr, ast.Tuple(list(map(ast.Constant, shape))))

    def valid(self, axes: tuple[gem.Index, ...]) -> ast.expr | None:
        """The points of `axes` that exist, or `None` if all of them do.

        The others are items past the last one, and the padding of an axis to a power of two.
        """
        terms = []
        for axis in axes:
            if axis == self._item:
                bound = ast.Name("num_items")
            elif (extent := static_extent(axis)) != self.size(axis):
                bound = ast.Constant(extent)
            else:
                continue
            terms.append(ast.Compare(self.expand(self._axes[axis], axes), [ast.Lt()], [bound]))
        return functools.reduce(lambda a, b: ast.BinOp(a, ast.BitAnd(), b), terms) if terms else None

    def index(self, entry: Any) -> Tile:
        """The tile of an entry of a multi-index"""
        if isinstance(entry, numbers.Integral):
            return Tile(ast.Constant(int(entry)))
        return self.process(entry.expression if isinstance(entry, gem.VariableIndex) else entry)

    def address(self, access: gem.Indexed, *others: Tile) -> tuple[ast.expr, ast.Tuple, tuple]:
        """Return the array, the index tiles and the axes of an access to a variable.

        The axes of `others` are also broadcast over.
        """
        variable, = access.children
        indices = [self.index(entry) for entry in access.multiindex]
        axes = self.union(*indices, *others)
        return ast.Name(variable.name), ast.Tuple([self.expand(i, axes) for i in indices]), axes

    # }}}

    def assign(self, instruction) -> None:
        """Store the value of `instruction.expression` at each point that exists.

        The first index of every other point is replaced by -1, and cuTile does not
        access an entry that is out of bounds.
        """
        assignee = instruction.assignee
        if not (isinstance(assignee, gem.Indexed) and isinstance(assignee.children[0], gem.Variable)):
            raise NotImplementedError(f"Assignment to {assignee!r} is not supported")

        value = self.process(instruction.expression)
        array, indices, axes = self.address(assignee, value)
        if self._item not in axes:
            raise NotImplementedError(f"{instruction!r} does not depend on the items")

        if (valid := self.valid(axes)) is not None:
            indices.elts[0] = ct("where", valid, indices.elts[0], ast.Constant(-1))
        indices = self.bind("at", indices).expr
        update = self.expand(value, axes)

        match instruction.mode:
            case "write":
                store = ct("scatter", array, indices, update)
            case "inc" if needs_atomic(instruction, (self._item,)):
                store = ct("atomic_add", array, indices, update)
            case "inc":
                store = ct("scatter", array, indices, ast.BinOp(ct("gather", array, indices), ast.Add(), update))
            case _:
                raise NotImplementedError(f"Assignment mode {instruction.mode!r} is not supported")
        self._body.append(ast.Expr(store))

    def process(self, node: gem.Node) -> Tile:
        """Return the tile of `node`, which is computed once for each instruction"""
        if node not in self._values:
            tile = self._process(node)
            # A constant is cheaper to repeat than to name
            if not isinstance(tile.expr, (ast.Constant, ast.Name)):
                tile = self.bind("v", tile.expr, tile.axes)
            self._values[node] = tile
        return self._values[node]

    @functools.singledispatchmethod
    def _process(self, node: Any) -> Tile:
        utils.raise_missing_dispatch_handler(node)

    @_process.register(gem.Index)
    def _(self, index: gem.Index) -> Tile:
        if index not in self._axes:
            self._order[index] = len(self._order)
            self._axes[index] = self.bind("i", self.arange(self.size(index)), (index,))
        return self._axes[index]

    @_process.register(gem.VariableIndex)
    def _(self, node: gem.VariableIndex) -> Tile:
        return self.process(node.expression)

    @_process.register(gem.Zero)
    @_process.register(gem.Literal)
    def _(self, node: gem.Constant) -> Tile:
        assert not node.shape
        if isinstance(node, gem.Zero):
            return Tile(ast.Constant(0.0))
        value = node.array[()]
        return Tile(ast.Constant(int(value)) if value.dtype.kind in "iu" else self.constant(value))

    @_process.register(gem.Indexed)
    def _(self, node: gem.Indexed) -> Tile:
        aggregate, = node.children
        if isinstance(aggregate, gem.Literal):
            return self.table(aggregate.array, node.multiindex)
        if not isinstance(aggregate, gem.Variable):
            raise NotImplementedError(f"Indexing {type(aggregate).__name__} is not supported")

        array, indices, axes = self.address(node)
        value = ct("gather", array, indices)
        # Integers are indices, which are all computed in 64 bits
        if np.dtype(aggregate.dtype).kind in "iu":
            value = ct("astype", value, ct("int64"))
        return Tile(value, axes)

    def table(self, array: np.ndarray, multiindex: tuple) -> Tile:
        """A constant tile for a tensor-valued literal, with zeros in the padding of each axis"""
        array = array[tuple(i if isinstance(i, numbers.Integral) else slice(None) for i in multiindex)]
        indices = [i for i in multiindex if not isinstance(i, numbers.Integral)]
        if not all(isinstance(i, gem.Index) for i in indices) or len(set(indices)) != len(indices):
            raise NotImplementedError(f"A literal indexed by {multiindex!r} is not supported")

        axes = self.union(*map(self.process, indices))
        array = array.transpose([indices.index(a) for a in axes])
        array = np.pad(array, [(0, self.size(a) - n) for a, n in zip(axes, array.shape)])
        return Tile(ct("astile", self.constant(nested(array)), dtype=ct(np.dtype(array.dtype).name)), axes)

    @_process.register(gem.IndexSum)
    def _(self, node: gem.IndexSum) -> Tile:
        body = self.process(node.children[0])
        summed = [i for i in node.multiindex if i in body.axes]
        value = body.expr
        if summed:
            summed_axes = self.union(*(self._axes[i] for i in summed))
            if (valid := self.valid(summed_axes)) is not None:
                valid = self.expand(Tile(valid, summed_axes), body.axes)
                value = ct("where", valid, value, ast.Constant(0.0))
            positions = ast.Tuple([ast.Constant(body.axes.index(i)) for i in summed])
            value = ct("sum", value, axis=positions)
        # The body is constant along the other summed indices
        for index in set(node.multiindex) - set(summed):
            value = ast.BinOp(value, ast.Mult(), ast.Constant(float(static_extent(index))))
        return Tile(value, tuple(a for a in body.axes if a not in summed))

    def elementwise(self, operands: tuple[gem.Node, ...]) -> tuple[list[ast.expr], tuple[gem.Index, ...]]:
        """Return the tiles of `operands`, reshaped so that they broadcast, and their axes"""
        tiles = [self.process(operand) for operand in operands]
        axes = self.union(*tiles)
        return [self.expand(tile, axes) for tile in tiles], axes

    _operators = {
        gem.Sum: ast.Add,
        gem.Product: ast.Mult,
        gem.Division: ast.Div,
        gem.FloorDiv: ast.FloorDiv,
        gem.Remainder: ast.Mod,
        gem.LogicalAnd: ast.BitAnd,
        gem.LogicalOr: ast.BitOr,
    }

    _functions = {gem.MinValue: "minimum", gem.MaxValue: "maximum", gem.Power: "pow", gem.Conditional: "where"}

    @_process.register(gem.Scalar)
    def _(self, node: gem.Scalar) -> Tile:
        operands, axes = self.elementwise(node.children)
        if type(node) in self._operators:
            return Tile(ast.BinOp(operands[0], self._operators[type(node)](), operands[1]), axes)
        if type(node) in self._functions:
            return Tile(ct(self._functions[type(node)], *operands), axes)
        utils.raise_missing_dispatch_handler(node)

    _comparisons = {"<": ast.Lt, "<=": ast.LtE, ">": ast.Gt, ">=": ast.GtE, "==": ast.Eq, "!=": ast.NotEq}

    @_process.register(gem.Comparison)
    def _(self, node: gem.Comparison) -> Tile:
        (a, b), axes = self.elementwise(node.children)
        return Tile(ast.Compare(a, [self._comparisons[node.operator]()], [b]), axes)

    # The GEM math functions that cuTile has under another name
    _math_names = {"ln": "log"}

    @_process.register(gem.MathFunction)
    def _(self, node: gem.MathFunction) -> Tile:
        operands, axes = self.elementwise(node.children)
        return Tile(ct(self._math_names.get(node.name, node.name), *operands), axes)


class TileExecutable(GemExecutable):
    """A kernel that cuTile compiles when it is first launched"""

    @cached_property
    def _item(self) -> gem.Index:
        if len(self.parallel_indices) != 1:
            raise NotImplementedError("A cuTile kernel has exactly one parallel loop")
        item, = self.parallel_indices
        return item

    @cached_property
    def _assignments(self) -> list:
        """The instructions without component tensors, as in `GemExecutable._impero`"""
        return [
            impero.Assignment(
                *impero_utils.preprocess_gem([insn.assignee]),
                *impero_utils.preprocess_gem([insn.expression]),
                insn.mode,
            )
            for insn in self.instructions
        ]

    @cached_property
    def _extent_variables(self) -> dict[str, gem.Variable]:
        """The kernel arguments that give the number of items, which is read on the host"""
        return variables(self._item.extent)

    @cached_property
    def _argument_names(self) -> list[str]:
        return [name for name, view in self.arguments.items() if not isinstance(view.buffer, NullBuffer)]

    @cached_property
    def argument_devices(self) -> list:
        device = get_current_device()
        return [HOST_DEVICE if name in self._extent_variables else device for name in self._argument_names]

    def as_callable_arg(self, buffer_view: Any, /, index: int | None = None) -> Any:
        with offloading(self.argument_devices[index]):
            return buffer_view.handle.reshape(-1)

    @cached_property
    def _callable(self):
        import cuda.tile as ct

        exprs = [e for insn in self._assignments for e in (insn.assignee, insn.expression)]
        # Each temporary is an array with one row per item, as `GemCompiler` declares it
        temporaries = {name: v for name, v in variables(*exprs).items() if is_temporary(v)}
        on_device = [name for name in self._argument_names if name not in self._extent_variables]

        # The largest tile has one entry per item of the block and per point of the other free indices of a node
        points = max(
            np.prod([padded(static_extent(i)) for i in node.free_indices if i != self._item], dtype=int)
            for node in pre_traversal(exprs)
        )
        items_per_block = int(min(MAX_ITEMS_PER_BLOCK, max(MAX_TILE_SIZE // points, 1)))
        builder = TileBuilder(self._assignments, self._item, items_per_block)
        source = builder.build([*on_device, *temporaries])

        # cuTile reads the source of the kernel through `inspect`, which looks in `linecache`
        filename = f"<pyop3-tile-{hashlib.sha1(source.encode()).hexdigest()[:12]}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        namespace = {"ct": ct, **builder.constants}
        exec(compile(source, filename, "exec"), namespace)
        kernel = ct.kernel(namespace["tile_kernel"])

        @functools.lru_cache(maxsize=1)
        def scratch(num_items: int) -> list:
            return [cp.empty((num_items, *map(int, v.shape[1:])), dtype=v.dtype) for v in temporaries.values()]

        def launch(*args) -> None:
            arrays = dict(zip(self._argument_names, args, strict=True))
            bindings = {v: arrays[name] for name, v in self._extent_variables.items()}
            extent, = interpreter.evaluate([gem.as_gem(self._item.extent)], bindings)
            num_items = int(extent.arr)

            grid = (max(utils.ceildiv(num_items, items_per_block), 1), 1, 1)
            kernel_args = (num_items, *(arrays[name] for name in on_device), *scratch(num_items))
            ct.launch(cp.cuda.get_current_stream(), grid, kernel, kernel_args)

        return launch
