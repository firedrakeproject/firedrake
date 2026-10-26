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
from pyop3.device import Device, HOST_DEVICE, get_current_device

from gem import gem, impero as imp
from gem.node import pre_traversal

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
    """Return a process handler for a GEM node with two scalar children.

    :arg float_op:  MLIR operation for floating point operand. Use None if op only defined for integers
    :arg int_op: MLIR operation for integer operand
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
        """ Key should only be defined once """
        assert any(key not in scope for scope in self._scopes)
        self._scopes[-1][key] = value

    def update(self, key, value: SSAValue) -> None:
        """ SSA value of a key is updated - used for summation operations
        
        Key should already exist 
        """
        assert any(key in scope for scope in self._scopes) 
        for scope in reversed(self._scopes):
            if key in scope:
                scope[key] = value
                break

    def __getitem__(self, key) -> SSAValue:
        """ Seek SSAValue from innermost -> outermost scope """
        for scope in reversed(self._scopes):
            if key in scope:
                return scope[key]
        raise KeyError(f"{key} is not in the SSA symbol table")

    def __contains__(self, key) -> bool:
        return any(key in s for s in self._scopes)


class MLIRBuilder:
    """ Class traverses an Impero AST and builds an MLIR Module Operation
    """
    def __init__(self) -> None:
        # If we can know function arguments in advance, we can create FuncOp as opposed to Block
        self._entry_block = Block()

        # Builder stack - Builder object acts as bridge to generate code for a given reg/block/op
        self._builder_stack: list[Builder] = [
            Builder(InsertPoint.at_end(self._entry_block))
        ]

        # Index into the builder stack of the region that `insert_at_entry` targets
        self._entry_depth: int = 0

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

        # Argument name -> Device that the argument must be placed on
        self._argument_devices: dict[str, Device] = dict()

    @property
    def builder(self) -> Builder:
        return self._builder_stack[-1]

    @property
    def arguments(self) -> list[Argument]:
        return sorted(self._arguments, key=lambda arg: arg.name)

    @property
    def argument_devices(self) -> dict[str, Device]:
        """The device that each function argument must be placed on.
        
        The default of the function argument is given by `get_current_device`
        (The device as of when generating takes place).

        """
        return self._argument_devices

    def build(self, root, name: str = "imp_kernel") -> tuple[ModuleOp, list[Argument], str]:
        """Construct the MLIR module, containing a FuncOp, given an Impero tree.

        :arg root: The Impero tree.
        :arg name: The name of the generated MLIR kernel.

        :returns tuple
            The module, the kernel arguments (sorted by name), and the name of the C
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

        # Update the return values to match similar to Loopy 
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
        
        top_builder = self._builder_stack[self._entry_depth]
        block = top_builder.insertion_point.block

        top_builder.insert(op, InsertPoint.at_start(block))
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
        """Apply float_op or int_op to a and b after promoting them to a common type.

        float_op can be None for operations that are only defined for integers.
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
        """Return the SSA value of the buffer that backs `node`, declaring it if needed.

        A `gem.Variable` becomes either a temporary allocation or a function
        argument, and a `gem.Literal` becomes a read-only global constant.
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

        """
        mlir_type = get_mlir_type(node.dtype)

        # NOTE: This is not nice. How else do I differ between temps and func args? 
        if node.data == "null":
            # ASSUMPTION: In temporary shape, first element is cell loop, second is actual shape.
            shape = [node.shape[1]]  
            return self._declare_global(
                node.name, mlir_type, np.zeros(shape=shape, dtype=node.dtype), constant=False
            )

        elif isinstance(node.data.buffer, ArrayBuffer):
            shape = [DYNAMIC_INDEX]
            memref_type = MemRefType(mlir_type, shape=shape)
            arg = self._entry_block.insert_arg(memref_type, len(self._entry_block.args))
            arg.name_hint = node.name

            self._arguments.append(Argument(node.name, node.dtype, shape))
            self._argument_devices.setdefault(node.name, get_current_device())
            return arg

        raise NotImplementedError(f"Unsupported buffer type: {type(node.data.buffer).__name__}")

    @_declare_buffer.register(gem.Literal)
    def _(self, node: gem.Literal) -> SSAValue:
        """ Define Literal (tensor-valued constant) as global constant read-only (flat) buffer """
        return self._declare_global(
            self.unique_name("literal"),
            get_mlir_type(node.dtype),
            node.array.flatten(),
            constant=True,
        )

    def _declare_global(
        self, name: str, mlir_type, array: np.ndarray, *, constant: bool
    ) -> SSAValue:
        """Declare a private module-level memref.global and fetch it in the entry block.

        The fetch is placed in the entry block so that it dominates all uses.

        :arg name: symbol name of the global.
        :arg mlir_type: MLIR element type.
        :arg array: A flat array that provides the initial contents and the shape.
        :arg constant: Boolean flag on whether the global is read-only.

        :raises ValueError: If a global with the same name is already declared.

        """
        # Make sure that existing global op does not exist
        if any(op.sym_name.data == name for op in self.global_ops):
            raise ValueError(f"A global with the name {name!r} is already declared")

        # TODO: Assign memory-level in the MemRefType
        memref_type = MemRefType(mlir_type, array.shape)
        value = DenseIntOrFPElementsAttr.from_list(
            TensorType(mlir_type, array.shape), array.data
        )

        self.global_ops.append(
            memref.GlobalOp.get(
                StringAttr(name),
                memref_type,
                value,
                sym_visibility=StringAttr("private"),
                constant=UnitAttr() if constant else None,
            )
        )

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
        """ Receives a list of ints and gem.{Index, VariableIndex} and returns SSA linearised           access 
        """

        assert len(multiindex) == len(strides), "Strides length must match indices"

        # linearised = sum(index * stride)
        linearised = None
        for ind, step in zip(multiindex, strides):
            term = self.get_index_ssa(ind)
            # Ignore factor of 1, constant folding 
            if step != 1:
                term = self.insert(arith.MuliOp(term, self._const(int(step), INDEX_TYPE)))
            linearised = term if linearised is None else self.insert(arith.AddiOp(linearised, term))

        return linearised if linearised is not None else self._const(0, INDEX_TYPE)

    def _address(self, node) -> tuple[SSAValue, list[SSAValue]]:
        """Return the buffer and the (flat) index """
        # TODO: Figure out why this assert is actually true
        variable = node.children[0]
        assert isinstance(variable, gem.Variable)

        multiindex = node.multiindex[0] if len(node.multiindex) == 1 else node.multiindex[1]
        index_ssa = self.get_index_ssa(multiindex) 
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
        strides = row_major_strides([mi.extent for mi in multiindex])
        return self._load(aggregate, multiindex, strides)

    @index_into.register(gem.Variable)
    def _(self, aggregate, multiindex) -> SSAValue:
        # FIXME: There is a bug here. Probably unsurprisingly
        # ASSUMPTION: First element is index is cell loop
        if len(multiindex) > 1: 
            multiindex = multiindex[1:] 

        return self._load(aggregate, multiindex, [1])

    # }}}

    def process(self, obj: Any, /, **kwargs):
        """Return the SSA value of `obj`, reusing a value that is already bound in scope.

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
        for child in tree.children:
            self.process(child)

    @_process.register(imp.For)
    def process_for(self, tree):
        assert tree.index.extent

        # NOTE: Index hopefully tagged with parallel at this stage
        # if tree.index.is_parallel...

        accumulators = self.find_accumulators(tree.children[0])

        with self.enter_for(tree.index, accumulators=accumulators):
            self.process(tree.children[0])

    # NOTE: I think I can clean this up. Can I use pre_traversal instead of walk?  
    # Only got walk because of the Impero nodes. 
    def find_accumulators(self, tree) -> list[gem.IndexSum]:
        """Return the IndexSums that are accumulated within `tree` but initialised outside of it.
        MLIR is an SSA compiler. Any scalar sums must be passed as iterator arguments.

        Given Impero AST construction, we need to seek any potential args 
        BEFORE entering loop.
        """
        accumulated = dict()
        initialised = set()

        def walk(node):
            if isinstance(node, imp.Accumulate):
                accumulated.setdefault(node.indexsum, None)
            elif isinstance(node, imp.Initialise):
                initialised.add(node.indexsum)
            elif isinstance(node, (imp.Block, imp.For)):
                for child in node.children:
                    walk(child)

        walk(tree)
        carried = [s for s in accumulated if s not in initialised]
        assert all(s in self.symbol_table for s in carried), "IndexSum has not been instantiated"
        return carried

    @_process.register(imp.Evaluate)
    def process_evaluate(self, leaf):
        """ 
        Calculate value within expression and assign to temporary 

        I feel that we don't need this... Determine reason and revisit. 
        """
        return

        # expr = leaf.expression
        # if expr not in self.symbol_table:
        #     self.symbol_table.define(expr, self.process(expr))
        # return self.symbol_table[expr]

    @_process.register(imp.Initialise)
    def process_initialise(self, leaf):
        """ Bind IndexSum to zero, the initial value for summation 

        Every loop that accumulates into the sum carries this value as an iter_arg.

        """
        indexsum = leaf.indexsum
        assert not indexsum.shape

        zero = self._const(0, get_mlir_type(indexsum.dtype))
        self.symbol_table.define(indexsum, zero)

    @_process.register(imp.Accumulate)
    def process_accumulate(self, leaf):
        """Add terms to scalar gem.IndexSum, and rebind the sum to the result.

        """
        indexsum = leaf.indexsum

        assert indexsum in self.symbol_table, "IndexSum has not been instantiated"
        # Inside the summation loop(s) this is the iter_arg of the innermost scf.for
        acc = self.symbol_table[indexsum]

        # The summed indices are bound by the enclosing For loops.
        term = self.process(indexsum.children[0])
        
        if not acc.type == term.type:
            raise TypeError(f"Accumulator and Summee should match type ({acc.type}, {term.type})")
        summed = self._binary(acc, term, arith.AddfOp, arith.AddiOp)

        self.symbol_table.update(indexsum, summed)
        return summed

    @_process.register(imp.Assignment)
    def process_assignment(self, leaf):
        """Store the value of `leaf.expression` into the buffer of `leaf.assignee`.

        For mode 'inc' the current value is loaded and the expression is added to it.
        """
        assignee = leaf.assignee
        if not isinstance(assignee, gem.Indexed):
            raise NotImplementedError(
                f"Assignment to {type(assignee).__name__} is not supported, expected gem.Indexed"
            )

        buf, idx = self._address(assignee)
        value = self.process(leaf.expression)

        if not buf.type.element_type == value.type:
            raise TypeError(f"Buffer and Value types do not match: {buf.type.element_type}, {value.type}")

        match leaf.mode:
            case "write":
                pass
            case "inc":
                current = self.insert(memref.LoadOp.get(buf, idx))
                value = self._binary(current, value, arith.AddfOp, arith.AddiOp)
            case _:
                raise NotImplementedError(f"Assignment mode {leaf.mode!r} is not supported")

        self.insert(memref.StoreOp.get(value, buf, idx))

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
        # A bound index is returned by the symbol table lookup in `process`.
        raise ValueError(f"Index {leaf!r} is used outside of a loop that binds it")

    @_process.register(gem.VariableIndex)
    def _(self, leaf):
        return self.process(leaf.expression)

    @_process.register(gem.Indexed)
    def _(self, leaf):
        return self.index_into(leaf.children[0], leaf.multiindex)
    
    @_process.register(gem.IndexSum)
    def process_indexsum(self, leaf):
        """
        IndexSum should only be processed through imp.Accumulate
        
        If we are reaching this point, we have not cached the result correctly
        """
        raise ValueError("IndexSum should be processed via imp.Accumulate, the result is failing to cache if here.")

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

    def declare_runtime_expr_on_host(self, expr) -> None:
        """ Mapping all function arguments related to `expr` onto the HOST_DEVICE, for runtime-known extent

        expr is possibly: (numbers.Integral, gem.VariableIndex)
        """
        if isinstance(expr, gem.VariableIndex):
            expr = expr.expression
        if not isinstance(expr, gem.Node):
            return

        # pre_traversal guarantees walk of the index expressions of gem.Indexed nodes
        for node in pre_traversal([expr]):
            # If variable and non-temporary 
            if isinstance(node, gem.Variable) and isinstance(node.data.buffer, ArrayBuffer):
                self._argument_devices[node.name] = HOST_DEVICE

    # TODO: Can we move this to ImplicitBuilder?
    @contextlib.contextmanager
    def enter_for(self, index, accumulators=()):
        """Open an `scf.for` loop over index

        :arg index: numbers.Integral or gem.Index, denoting extent of domain
        :arg accumulators: keys in the symbol table (e.g. gem.IndexSum) whose values are
            carried through the loop as iter_args. Inside the loop they are bound to the
            block arguments, and after the loop they are rebound to the loop results.
        """
        assert isinstance(index, (numbers.Integral, gem.Index))

        extent = index if isinstance(index, numbers.Integral) else index.extent

        # Loop bounds must be read on host and used for setting GPU grid bounds
        self.declare_runtime_expr_on_host(extent)

        lb = self._const(0, INDEX_TYPE)
        ub = self.get_index_ssa(extent)
        step = self._const(1, INDEX_TYPE)

        init = [self.symbol_table[key] for key in accumulators]
        arg_types = [INDEX_TYPE, *(value.type for value in init)]

        loop_op = scf.ForOp(lb, ub, step, init, Region(Block(arg_types=arg_types)))
        self.insert(loop_op)

        body = loop_op.body.block
        self.symbol_table.push()
        self.symbol_table.define(index, body.args[0])

        for key, arg in zip(accumulators, body.args[1:]):
            self.symbol_table.define(key, arg)

        self._builder_stack.append(Builder(InsertPoint.at_end(body)))

        yield

        self.insert(scf.YieldOp(*(self.symbol_table[key] for key in accumulators)))
        self._builder_stack.pop()
        self.symbol_table.pop()

        # Include the summed arguments into the scope upon exit  
        for key, result in zip(accumulators, loop_op.results):
            self.symbol_table.update(key, result)

    @contextlib.contextmanager
    def enter_parallel(self, index, is_outermost=False):
        """Open an `scf.parallel` loop over index

        :arg index: numbers.Integral or gem.Index, denoting extent of domain
        :arg is_outermost: boolean flag which defines if this loop should be considered outermost loop for GPU

        """
        # TODO: Add GPU mapping dimensions
        assert isinstance(index, (numbers.Integral, gem.Index))

        extent = index if isinstance(index, numbers.Integral) else index.extent

        # Loop bounds must be read on host and used for setting GPU grid bounds
        self.declare_runtime_expr_on_host(extent)

        lb = self._const(0, INDEX_TYPE)
        ub = self.get_index_ssa(extent)
        step = self._const(1, INDEX_TYPE)

        loop_op = scf.ParallelOp([lb], [ub], [step], Region(Block(arg_types=[INDEX_TYPE])))
        self.insert(loop_op)

        body = loop_op.body.block
        self.symbol_table.push()
        self.symbol_table.define(index, body.args[0])
        self._builder_stack.append(Builder(InsertPoint.at_end(body)))

        yield

        self.insert(scf.ReduceOp())
        self._builder_stack.pop()
        self.symbol_table.pop()

    @contextlib.contextmanager
    def outermost(self):
        """ Increments the visible outermost scope of the insertion problem, while inside the context

        This is what will be used for entering a GPU launch context - all CPU components outside it should not be accessible. 

        The current outermost region, and the SSA values that are defined in it,
        become invisible to everything inside the context. Operations that are
        hoisted with insert_at_entry are inserted one level deeper instead.

        The constant and buffer caches are emptied while inside the context,
        because their SSA values are not valid there, and they are restored on
        exit. Funtcion arguments are not reset because they are valid across 
        scope (passed as CPU/GPU arguments as per device mapping) 

        """
        original_scopes = self.symbol_table._scopes
        original_constants = self._constants
        original_buffers = self._buffers

        argument_names = {arg.name for arg in self._arguments}
        self.symbol_table._scopes = original_scopes[1:]
        self._constants = dict()
        self._buffers = {
            key: ssa for key, ssa in original_buffers.items() if key in argument_names
        }
        self._entry_depth += 1

        yield

        self._entry_depth -= 1
        self.symbol_table._scopes = [original_scopes[0]] + self.symbol_table._scopes
        self._constants = original_constants

        # Arguments that were first used inside the context must stay declared
        argument_names = {arg.name for arg in self._arguments}
        original_buffers.update(
            {key: ssa for key, ssa in self._buffers.items() if key in argument_names}
        )
        self._buffers = original_buffers 


def row_major_strides(shape):
    strides = []
    acc = 1
    for extent in reversed(shape):
        strides.append(acc)
        acc *= extent
    return strides[::-1]
