from __future__ import annotations

import functools
import contextlib
import dataclasses
import numbers
import types
import numpy as np
from typing import Any

from pyop3 import utils
from pyop3.buffer import NullBuffer, ArrayBuffer

from gem import gem, impero as imp

from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects import math as xmath
from xdsl.builder import Builder, InsertPoint
from xdsl.ir import SSAValue, Block, Region, Operation

from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    DYNAMIC_INDEX,
    IntegerType,
    IndexType,
    IntegerAttr,
    FloatAttr,
    i1, i64, f32, f64,
    MemRefType,
    ModuleOp,
    StringAttr,
    TensorType,
    UnitAttr,
)

INDEX_TYPE = IndexType()
FLOAT_TYPES = (f32, f64)


@dataclasses.dataclass
class Argument:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...] | None

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{self.name}, dtype: {self.dtype}, shape: {self.shape or '?'}>"


def get_mlir_type(dtype):
    # NOTE: Default to f64 if dtype is None, possibly wrong.
    if dtype is None:
        return f64

    dt = np.dtype(dtype)
    if dt.kind == "f":
        return f64 if dt.itemsize == 8 else f32
    elif dt.kind in "iu":
        return IntegerType(8 * dt.itemsize)
    elif dt.kind == "b":
        return i1

    raise NotImplementedError(f"No MLIR element type for dtype {dt}")

def is_float(t) -> bool:
    return t in FLOAT_TYPES

# NOTE: Do I want this function? It feels wrong grouping these types
def is_intlike(t) -> bool:
    return t == INDEX_TYPE or isinstance(t, IntegerType)


def _binary_handler(float_op, int_op):
    """Return a ``process`` handler for a GEM node with two scalar children.

    Parameters
    ----------
    float_op, int_op :
        The MLIR operations to use for floating point and integer operands.
        Pass ``None`` for ``float_op`` if the operation is only defined for
        integers.

    """
    def handler(self, expr):
        a, b = (self.process(c) for c in expr.children)
        return self._binary(a, b, float_op, int_op)
    return handler


class SymbolTable:
    """
    Class deals with SSAValues for a respective scope.

    e.g: SSAValues inside scf.ForOp are not valid outside the iterative loop.
    """
    def __init__(self) -> None:
        self._scopes: list[dict[Any, SSAValue]] = [dict()]

    def push(self) -> None:
        self._scopes.append(dict())

    def pop(self) -> None:
        self._scopes.pop()

    def define(self, key, value: SSAValue) -> None:
        """ Key should not be re-defined. SSA only permits single definition """
        assert key not in self._scopes[-1]
        self._scopes[-1][key] = value

    def __getitem__(self, key) -> SSAValue:
        """ Seek SSAValue from innermost -> outermost scope """
        for scope in reversed(self._scopes):
            if key in scope:
                return scope[key]
        raise KeyError(f"{key} is not in the SSA symbol table")

    def __contains__(self, key) -> bool:
        return any(key in s for s in self._scopes)


class MLIRBuilder:
    """Visitor that populates an MLIR Object.

    This class is intended to be subclassed by 'reconstruction' visitors that
    build similar objects.

    """
    def __init__(self) -> None:
        # If we can know function arguments in advance, we can create FuncOp as opposed to Block
        self._entry_block = Block()

        # Builder stack - Builder object acts as bridge to generate code for a given reg/block/op
        self._builder_stack: list[Builder] = [
            Builder(InsertPoint.at_end(self._entry_block))
        ]

        # Symbol table (inames/loop-idx and evaluated expressions)
        self.symbol_table = SymbolTable()

        # Variable name (or gem.Literal) -> SSAValue of its buffer
        self._buffers: dict[Any, SSAValue] = dict()

        # Constant attribute -> SSAValue of its arith.constant in the entry block
        self._constants: dict[FloatAttr | IntegerAttr, SSAValue] = dict()

        # Globally defined constants - like quadrature weights
        self.global_ops: list[Operation] = list()

        self._name_generator = utils.UniqueNameGenerator()

        # Arguments to return to buffer
        self._arguments: list[Argument] = list()

    @property
    def builder(self) -> Builder:
        return self._builder_stack[-1]

    @property
    def arguments(self) -> list[Argument]:
        return sorted(self._arguments, key=lambda arg: arg.name)

    def build(self, root, name: str = "imp_kernel") -> tuple[ModuleOp, list[Argument], str]:
        """Construct the MLIR module, containing a FuncOp, given an Impero tree.

        Parameters
        ----------
        root :
            The Impero tree.
        name :
            The name of the generated MLIR kernel.

        Returns
        -------
        tuple
            The module, the kernel arguments (sorted by name, which is also
            the order of the function parameters), and the name of the C
            interface symbol.

        """
    
        self.process(root)
        n = len(self._arguments)

        perm = sorted(range(n), key=lambda i: self._arguments[i].name)
        arg_types = [self._entry_block.args[i].type for i in perm]

        func_op = func.FuncOp(name, (arg_types, ()))
        func_op.attributes["llvm.emit_c_interface"] = UnitAttr()
        new_block = func_op.body.block

        for new_index, old_index in enumerate(perm):
            old_arg = self._entry_block.args[old_index]
            new_arg = new_block.args[new_index]
            old_arg.replace_all_uses_with(new_arg)

        ops = list(self._entry_block.ops)
        for op in ops:
            op.detach()
        new_block.add_ops(ops)
        new_block.add_op(func.ReturnOp())

        modop = ModuleOp([*self.global_ops, func_op])

        modop.verify()

        return modop, self.arguments, f"_mlir_ciface_{name}"

    # {{{ helpers

    def unique_name(self, prefix: str) -> str:
        return self._name_generator(prefix)

    def insert(self, op: Operation) -> SSAValue | None:
        """Insert an MLIR operation into the current block, returning its first result."""
        self.builder.insert(op)
        results = op.results
        return results[0] if results else None

    def insert_at_entry(self, op: Operation) -> SSAValue:
        """Insert an MLIR operation at the start of the entry block.

        Values that are defined here dominate every loop nest.

        """
        Builder(InsertPoint.at_start(self._entry_block)).insert(op)
        return op.results[0]

    def _const(self, value, mlir_type) -> SSAValue:
        """Return the SSA value of a scalar constant, reusing it if it already exists.

        Constants are defined in the entry block, so that they dominate every use.
        """
        if is_float(mlir_type):
            attr = FloatAttr(float(value), mlir_type)
        else:
            attr = IntegerAttr(int(value), mlir_type)
        if attr not in self._constants:
            self._constants[attr] = self.insert_at_entry(arith.ConstantOp(attr))
        return self._constants[attr]

    # NOTE: Switch this into dispatch function? It is super ugly but it is merely comparison
    # FIXME: Improve this later on. It forms the type conversion basis which is important.
    # I think it could look cleaenr for debugging
    def _convert_type(self, v: SSAValue, to) -> SSAValue:
        """ Function converts an SSA to the desired type `to`, if necessary

        Allows for:
          - Truncation or extension of values
          - Index casting
          - Float / Integer conversion

        Consider dropping the extension of values.
        """

        frm = v.type
        if frm == to:
            return v

        if is_intlike(frm) and is_intlike(to):
            if INDEX_TYPE in (frm, to):
                return self.insert(arith.IndexCastOp(v, to))
            if to.width.data > frm.width.data:
                return self.insert(arith.ExtSIOp(v, to))
            return self.insert(arith.TruncIOp(v, to))

        elif is_intlike(frm) and is_float(to):
            if frm == INDEX_TYPE:
                v = self.insert(arith.IndexCastOp(v, i64))
            return self.insert(arith.SIToFPOp(v, to))

        elif is_float(frm) and is_float(to):
            cls = arith.ExtFOp if to == f64 else arith.TruncFOp
            return self.insert(cls(v, to))

        elif is_float(frm) and is_intlike(to):
            v = self.insert(arith.FPToSIOp(v, i64 if to == INDEX_TYPE else to))
            return self._convert_type(v, to)

        raise NotImplementedError(f"Casting from {frm} to {to} is not implemented")

    def _to_index(self, ssa: SSAValue) -> SSAValue:
        return self._convert_type(ssa, INDEX_TYPE)

    def _align_scalars(self, a: SSAValue, b: SSAValue):
        """ Bring two scalars to common type

        Promotion order:
        (float > index > int)
        (wider bit > smaller bit)

        Index is ambiguous but if it needs to be stored in buffer, it will be recast to buffer type.

        """

        if a.type == b.type:
            return a, b

        if is_float(a.type) or is_float(b.type):
            floats = [t for t in (a.type, b.type) if is_float(t)]
            target = f64 if f64 in floats else floats[0]

        elif INDEX_TYPE in (a.type, b.type):
            target = INDEX_TYPE

        # NOTE: Really don't like promotion of types
        else:
            target = a.type if a.type.width.data >= b.type.width.data else b.type

        return self._convert_type(a, target), self._convert_type(b, target)

    def _binary(self, a, b, float_op, int_op) -> SSAValue:
        """Apply ``float_op`` or ``int_op`` to ``a`` and ``b`` after promoting them to a common type.

        ``float_op`` can be ``None`` for operations that are only defined for integers.
        """
        a, b = self._align_scalars(a, b)
        if is_float(a.type):
            if float_op is None:
                raise TypeError(f"{int_op.__name__} is not defined for floating point operands")
            return self.insert(float_op(a, b))
        return self.insert(int_op(a, b))

    # }}}

    # {{{ buffers

    def buffer_for(self, node) -> SSAValue:
        """Return the SSA value of the buffer that backs ``node``, declaring it if needed.

        A ``gem.Variable`` becomes either a temporary allocation or a function
        argument, and a ``gem.Literal`` becomes a read-only global constant.
        """
        key = node.name if isinstance(node, gem.Variable) else node
        if key not in self._buffers:
            buffer_ssa = self._declare_buffer(node)
            self._buffers[key] = buffer_ssa
        return self._buffers[key]

    @functools.singledispatchmethod
    def _declare_buffer(self, node) -> SSAValue:
        utils.raise_missing_dispatch_handler(node)

    @_declare_buffer.register(gem.Variable)
    def _(self, node: gem.Variable) -> SSAValue:
        """ Creates runtime-tensors

        Determining between function argument and temporary:
        -> isinstance(node.data.buffer, NullBuffer) -> temporary
        -> isinstance(node.data.buffer, ArrayBuffer) -> function argument

        TODO: Opportunity to tag buffers with memory space attributes here
        """
        # Assuming buffer is flat, we are doing one-form assembly anyway
        assert len(node.data.buffer.shape) == 1
        shape = list(map(int, node.data.buffer.shape))
        mlir_type = get_mlir_type(node.dtype)

        if isinstance(node.data.buffer, NullBuffer):
            # NOTE: AllocOps are used because AllocaOp is bad on GPU.
            # A compiler pass can raise Alloc to Alloca, and inserts the
            # deallocations that Alloc requires.
            # TODO: Temporaries are not zero-initialised.
            arg = self.insert_at_entry(memref.AllocOp.get(mlir_type, shape=shape))
            arg.name_hint = node.name 
            return arg 

        elif isinstance(node.data.buffer, ArrayBuffer):
            memref_type = MemRefType(mlir_type, shape=[DYNAMIC_INDEX])
            arg = self._entry_block.insert_arg(memref_type, len(self._entry_block.args))
            arg.name_hint = node.name

            self._arguments.append(Argument(node.name, node.dtype, shape))
            return arg

        raise NotImplementedError(f"Unsupported buffer type: {type(node.data.buffer).__name__}")

    @_declare_buffer.register(gem.Literal)
    def _(self, node: gem.Literal) -> SSAValue:
        """ Define Literal (tensor-valued constant) as global constant read-only (flat) buffer """
        flattened = node.array.flatten()
        mlir_type = get_mlir_type(node.dtype)
        memref_type = MemRefType(mlir_type, flattened.shape)

        name = self.unique_name("literal")

        value = DenseIntOrFPElementsAttr.from_list(
            TensorType(mlir_type, flattened.shape), flattened.data
        )

        self.global_ops.append(
            memref.GlobalOp.get(
                StringAttr(name),
                memref_type,
                value,
                sym_visibility=StringAttr("private"),
                constant=UnitAttr()
            )
        )

        # Fetch the global in the entry block so that it dominates all uses.
        return self.insert_at_entry(memref.GetGlobalOp(name, memref_type))

    # }}}

    # {{{ indexing

    @functools.singledispatchmethod
    def get_index_ssa(self, idx) -> SSAValue:
        utils.raise_missing_dispatch_handler(idx)

    @get_index_ssa.register(numbers.Integral)
    def _(self, idx: numbers.Integral):
        return self._const(idx, INDEX_TYPE)

    @get_index_ssa.register(gem.Constant)
    def _(self, idx: gem.Constant):
        return self._const(idx.value, INDEX_TYPE)

    @get_index_ssa.register(gem.VariableIndex)
    @get_index_ssa.register(gem.Index)
    @get_index_ssa.register(gem.Node)
    def _(self, idx):
        return self._to_index(self.process(idx))

    def linearise_index(self, strides, multiindex) -> SSAValue:
        """ Receives a list of ints and gem.{Index, VariableIndex} and returns SSA linearised access """
        assert len(multiindex) == len(strides), "Strides length must match indices"

        # linearised = sum(index * stride)
        linearised = None
        for ind, step in zip(multiindex, strides):
            term = self.get_index_ssa(ind)
            if step != 1:
                term = self.insert(arith.MuliOp(term, self._const(int(step), INDEX_TYPE)))
            linearised = term if linearised is None else self.insert(arith.AddiOp(linearised, term))

        return linearised if linearised is not None else self._const(0, INDEX_TYPE)

    def row_major_strides(self, shape):
        strides = []
        acc = 1
        for extent in reversed(shape):
            strides.append(acc)
            acc *= extent
        return strides[::-1]

    def _address(self, node) -> tuple[SSAValue, list[SSAValue]]:
        """Return the buffer and the (flat) index """
        # TODO: Figure out why this assert is actually true
        variable = node.children[0]
        assert isinstance(variable, gem.Variable)

        # TODO: In what scenario do we have more than one multiindex? 
        index_ssa = self.get_index_ssa(node.multiindex[0]) 
        return self.buffer_for(variable), [index_ssa]

    def _load(self, aggregate, multiindex, strides) -> SSAValue:
        buf = self.buffer_for(aggregate)
        return self.insert(
            memref.LoadOp.get(buf, [self.linearise_index(strides, multiindex)])
        )

    @functools.singledispatchmethod
    def index_into(self, aggregate, multiindex) -> SSAValue:
        """ Value of aggregate[multiindex] for any tensor-valued aggregate """
        utils.raise_missing_dispatch_handler(aggregate)

    @index_into.register(gem.Zero)
    def _(self, aggregate: gem.Zero, multiindex) -> SSAValue:
        return self._const(0, get_mlir_type(aggregate.dtype))

    @index_into.register(gem.Identity)
    def _(self, aggregate: gem.Identity, multiindex) -> SSAValue:
        i, j = (self.get_index_ssa(k) for k in multiindex)
        eq = self.insert(arith.CmpiOp(i, j, "eq"))
        t = get_mlir_type(aggregate.dtype)
        return self.insert(arith.SelectOp(eq, self._const(1, t), self._const(0, t)))

    @index_into.register(gem.Literal)
    def _(self, aggregate, multiindex) -> SSAValue:
        strides = self.row_major_strides([mi.extent for mi in multiindex])
        return self._load(aggregate, multiindex, strides)

    @index_into.register(gem.Variable)
    def _(self, aggregate, multiindex) -> SSAValue:
        # NOTE: This is while Connor debugs multiindices in GEM
        return self._load(aggregate, multiindex, [1])

    # }}}

    def process(self, obj: Any, /, **kwargs):
        """Return the SSA value of ``obj``, reusing a value that is already bound in scope.

        Only GEM nodes are looked up in the symbol table. A value that is
        defined inside a loop is visible only within that loop, so a later
        use outside the loop computes the value again.
        """
        if isinstance(obj, gem.Node) and obj in self.symbol_table:
            return self.symbol_table[obj]
        return self._process(obj, **kwargs)

    @functools.singledispatchmethod
    def _process(self, obj: Any, /, **kwargs):
        utils.raise_missing_dispatch_handler(obj)

    # {{{ impero types

    @_process.register(imp.Block)
    def process_block(self, tree):
        # FIXME: I am getting children in opposite order to intuition
        for child in tree.children[::-1]:
            self.process(child)

    @_process.register(imp.For)
    def process_for(self, tree):
        assert tree.index.extent

        # NOTE: Index hopefully tagged with parallel at this stage
        # if tree.index.is_parallel...

        with self.enter_for(tree.index):
            self.process(tree.children[0])

    @_process.register(imp.Return)
    def process_return(self, leaf):
        """ Store value of expression into variable

        If variable is Indexed - we store into the buffer
        If variable is not Indexed, we just store by name into symbol table

        """
        var = leaf.variable
        value = self.process(leaf.expression)

        if not isinstance(var, gem.Indexed):
            self.symbol_table.define(var.name, value)
            return

        buf, idx = self._address(var)
        self.insert(memref.StoreOp.get(value, buf, idx))

    @_process.register(imp.Evaluate)
    def process_evaluate(self, leaf):
        """ Calculate value within expression and assign to temporary """
        expr = leaf.expression
        if expr not in self.symbol_table:
            self.symbol_table.define(expr, self.process(expr))
        return self.symbol_table[expr]

    # }}}

    # {{{ gem types

    @_process.register(gem.Zero)
    def _(self, leaf):
        assert not leaf.shape
        return self._const(0, get_mlir_type(leaf.dtype))

    @_process.register(gem.Literal)
    def _(self, leaf):
        """ Process gem.Literal which is a scalar or tensor constant """
        if leaf.shape:
            return self.buffer_for(leaf) 
        else:
            return self._const(leaf.value, get_mlir_type(leaf.dtype))

    @_process.register(gem.Index)
    def _(self, leaf):
        # A bound index is returned by the symbol table lookup in ``process``.
        raise ValueError(f"Index {leaf!r} is used outside of a loop that binds it")

    @_process.register(gem.VariableIndex)
    def _(self, leaf):
        return self.process(leaf.expression)

    @_process.register(gem.Indexed)
    @_process.register(gem.Gather)
    def _(self, leaf):
        return self.index_into(leaf.children[0], leaf.multiindex)

    # }}}

    # {{{ gem operations

    # TODO: This is kind of ugly imo, but it is concise at least   
    _process.register(gem.Sum)(_binary_handler(arith.AddfOp, arith.AddiOp))
    _process.register(gem.Product)(_binary_handler(arith.MulfOp, arith.MuliOp))
    _process.register(gem.Division)(_binary_handler(arith.DivfOp, arith.DivSIOp))
    _process.register(gem.MinValue)(_binary_handler(arith.MinimumfOp, arith.MinSIOp))
    _process.register(gem.MaxValue)(_binary_handler(arith.MaximumfOp, arith.MaxSIOp))
    _process.register(gem.FloorDiv)(_binary_handler(None, arith.FloorDivSIOp))
    _process.register(gem.Remainder)(_binary_handler(None, arith.RemSIOp))
    _process.register(gem.LogicalAnd)(_binary_handler(None, arith.AndIOp))
    _process.register(gem.LogicalOr)(_binary_handler(None, arith.OrIOp))

    _CMPF = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge", "==": "oeq", "!=": "one"}
    _CMPI = {"<": "slt", "<=": "sle", ">": "sgt", ">=": "sge", "==": "eq", "!=": "ne"}

    @_process.register(gem.Comparison)
    def _(self, expr):
        a, b = self._align_scalars(*(self.process(c) for c in expr.children))
        if is_float(a.type):
            return self.insert(arith.CmpfOp(a, b, self._CMPF[expr.operator]))
        return self.insert(arith.CmpiOp(a, b, self._CMPI[expr.operator]))

    @_process.register(gem.Conditional)
    def _(self, expr):
        # NOTE: Worth noting that arith.select is eager
        # scf.if may be better if branch leads to div-by-zero
        cond, then, else_ = (self.process(c) for c in expr.children)
        then, else_ = self._align_scalars(then, else_)
        return self.insert(arith.SelectOp(cond, then, else_))

    # TODO: Need to deal with more possible functions
    @_process.register(gem.MathFunction)
    def _(self, expr):
        name = expr.name
        args = [self.process(c) for c in expr.children]
        if name == "abs":
            return self.insert((xmath.AbsFOp if is_float(args[0].type) else xmath.AbsIOp)(args[0]))
        raise NotImplementedError(f"Math function {name!r} is not implemented")

    # }}}

    # {{{ misc types

    @_process.register
    def _(self, obj: types.NoneType, /, **kwargs):
        return obj

    @_process.register
    def _(self, obj: numbers.Number, /, **kwargs):
        return self._const(obj, i64 if isinstance(obj, numbers.Integral) else f64)

    # }}}

    @contextlib.contextmanager
    def enter_for(self, index):
        """Open an ``scf.for`` loop over ``index`` (a ``gem.Index`` or an integer extent)."""
        extent = index if isinstance(index, numbers.Integral) else index.extent

        lb = self._const(0, INDEX_TYPE)
        ub = self.get_index_ssa(extent)
        step = self._const(1, INDEX_TYPE)

        for_op = scf.ForOp(lb, ub, step, [], Region(Block(arg_types=[INDEX_TYPE])))
        self.insert(for_op)

        body = for_op.body.block
        self.symbol_table.push()
        self.symbol_table.define(index, body.args[0])
        self._builder_stack.append(Builder(InsertPoint.at_end(body)))

        try:
            yield
            self.insert(scf.YieldOp())
        finally:
            self._builder_stack.pop()
            self.symbol_table.pop()
