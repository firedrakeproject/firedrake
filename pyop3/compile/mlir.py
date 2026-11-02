from __future__ import annotations

import functools
import contextlib
import numbers
import types
import numpy as np
from collections.abc import Mapping
from typing import Any

import pyop3.config 
from pyop3 import utils
from pyop3.buffer import NullBuffer, ArrayBuffer, IndexedBuffer
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

# Allow LLVM to fuse floating-point multiply-add. Only rounds once instead of twice though.
# Does not get other changes that -ffast-math normally applies for a GCC/clang compiler 
FP_CONTRACT = arith.FastMathFlagsAttr([arith.FastMathFlag.ALLOW_CONTRACT])

def get_mlir_type(dtype):
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

# TODO: Should this be somewhere else? This is only for GPU-case
def atomic_add_kind(t) -> int:
    """Return the `arith.AtomicRMWKind` value that adds two values of type `t`."""
    ATOMIC_ADDF, ATOMIC_ADDI = 0, 1
    return ATOMIC_ADDF if is_float(t) else ATOMIC_ADDI

# NOTE: Do I want this function? It feels wrong grouping these types
def is_intlike(t) -> bool:
    return t == INDEX_TYPE or isinstance(t, IntegerType)


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
    def __init__(
        self,
        root,
        arguments: Mapping[str, IndexedBuffer],
        name: str = "imp_kernel",
        *,
        temporary_indices: Mapping[gem.Node, tuple[gem.Index, ...]] | None = None,
    ) -> None:
        """Create the kernel FuncOp, with one memref argument for each buffer in `arguments`.

        :arg root: Impero tree that `build` translates.
        :arg arguments: Map from the kernel name of each buffer to its buffer view.
            Temporaries (views of a `NullBuffer`) are not function arguments and are dropped.
        :arg name: The name of the generated MLIR kernel
        :arg temporary_indices: `ImperoC.indices`. This maps each GEM temporary (the
            expression of an `imp.Evaluate`, or a `gem.IndexSum`) to the free indices
            that the loops around its declaration do not bind. A temporary that has
            such indices is read in a different loop from the one that computes it,
            so its values are stored in an array.

        """
        self._root = root

        # GEM temporary -> free indices that the loops around its declaration do not bind
        self._temporary_indices: Mapping[gem.Node, tuple[gem.Index, ...]] = (
            temporary_indices if temporary_indices is not None else {}
        )

        # GEM temporary -> SSA value of the array that stores its values
        self._temporaries: dict[gem.Node, SSAValue] = dict()

        # Kernel name -> buffer view, for the buffers that are function arguments
        # Excluding NullBuffers that are not function arguments 
        self._arguments: dict[str, IndexedBuffer] = {
            kernel_name: buffer_view
            for kernel_name, buffer_view in arguments.items()
            if not isinstance(buffer_view.buffer, NullBuffer)
        }

        # Collect MLIR memref types for FuncOp operation
        arg_types = [
            MemRefType(get_mlir_type(buffer_view.buffer.dtype), shape=[DYNAMIC_INDEX])
            for buffer_view in self._arguments.values()
        ]

        self._func_op = func.FuncOp(name, (arg_types, ()))
        self._func_op.attributes["llvm.emit_c_interface"] = UnitAttr()
        entry_block = self._func_op.body.block

        # Builder stack - Builder object acts as bridge to generate code for a given reg/block/op
        self._builder_stack: list[Builder] = [
            Builder(InsertPoint.at_end(entry_block))
        ]

        # Index into the builder stack of the region that `insert_at_entry` targets
        self._entry_depth: int = 0

        # Symbol table (inames/loop-idx and evaluated expressions)
        self.symbol_table = SymbolTable()

        # Variable name (or gem.Literal) -> SSAValue of its buffer
        self._buffers: dict[Any, SSAValue] = dict()

        # Constant attribute -> SSAValue of its arith.constant in the entry block
        self._constants: dict[FloatAttr | IntegerAttr, SSAValue] = dict()

        # Globally defined arrays - like quadrature weights
        self.global_ops: list[Operation] = list()

        self._name_generator = utils.UniqueNameGenerator()

        # Argument name -> Device that the argument must be placed on
        self._argument_devices: dict[str, Device] = dict.fromkeys(self._arguments, get_current_device())

        # Buffer mapping from buffer-name to SSA argument 
        self._buffers = dict(zip(self._arguments, entry_block.args))


    def to_string(self, operation: ModuleOp) -> str: 
        """ Returns Module operation as a string """ 
        from xdsl.printer import Printer 
        from io import StringIO
        output = StringIO()
        printer = Printer(stream=output)
        printer.print_op(operation)
        return output.getvalue()

    @property
    def builder(self) -> Builder:
        return self._builder_stack[-1]

    @property
    def _in_parallel_loop(self) -> bool:
        """Whether code is currently generated inside the body of the parallel loop."""
        return self._entry_depth > 0

    @property
    def arguments(self) -> dict[str, IndexedBuffer]:
        """ Arguments as pyop3 buffers """
        return [arg.buffer for arg in self._arguments.values()]

    @property
    def argument_devices(self) -> dict[str, Device]:
        """The device that each function argument must be placed on.
        
        The default of the function argument is given by `get_current_device`
        (The device as of when code generation takes place).
        
        Returns list of devices, order should match the exec arguments 

        """
        return list(self._argument_devices.values())

    def build(self) -> tuple[ModuleOp, dict[str, IndexedBuffer], str]:
        """Construct the MLIR module, containing the kernel FuncOp, from the Impero tree.

        :returns tuple (The module, the kernel arguments, ctype function name)
        """
        self.process(self._root)
        self.insert(func.ReturnOp())

        modop = ModuleOp([*self.global_ops, self._func_op])

        if pyop3.config.debug_checks:
            modop.verify()

        module_code = self.to_string(modop)

        return module_code, self.arguments, f"_mlir_ciface_{self._func_op.sym_name.data}"

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
        Floating-point operations carry the `contract` fast-math flag, so that LLVM can
        fuse a multiply and the add that uses its result into one FMA instruction.
        """
        a, b = self._align_scalars(a, b)
        if is_float(a.type):
            if float_op is None:
                raise TypeError(f"{int_op.__name__} is not defined for floating point operands")
            return self.insert(float_op(a, b, flags=FP_CONTRACT))
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
        """Declare a temporary as a stack allocation at the start of the entry block.

        Function arguments are declared when the builder is created, so only
        temporaries reach this point.

        Inside the parallel loop the entry block is the loop body, so each
        iteration (each GPU thread) gets its own copy of the temporary. The
        allocation is not initialised, so each entry must be written before it is read.

        """
        if node.data != "null":
            raise ValueError(f"{node.name!r} is neither a temporary nor a kernel argument")

        # FIXME: Avoiding attached cell loop
        shape = node.shape[-1:]
        return self.insert_at_entry(memref.AllocaOp.get(get_mlir_type(node.dtype), shape=shape))


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

    # TODO: I think {_temporary_offset, _store_temporary, _load_temporary} 
    # can be folded into existing functions
    def _temporary_offset(self, expr) -> SSAValue:
        """Return the flat position of the current entry of the array temporary of `expr`.

        The entry is given by the current values of the loop indices in
        `self._temporary_indices[expr]`, in row-major order.
        """
        indices = self._temporary_indices[expr]
        extents = [index.extent for index in indices]
        if not all(isinstance(extent, numbers.Integral) for extent in extents):
            raise NotImplementedError(f"Temporary arrays must have a compile-time size, not {extents}")
        return self.linearise_index(indices, row_major_strides(extents))

    def _store_temporary(self, expr, value: SSAValue) -> None:
        """Store `value` into the current entry of the array temporary of `expr`.

        The array is allocated on first use at the start of the entry block. Inside the
        parallel loop the entry block is the loop body, so each iteration (each GPU
        thread) gets its own copy.
        """
        if expr not in self._temporaries:
            size = int(np.prod([index.extent for index in self._temporary_indices[expr]]))
            self._temporaries[expr] = self.insert_at_entry(
                memref.AllocaOp.get(value.type, shape=[size])
            )
        offset = self._temporary_offset(expr)
        self.insert(memref.StoreOp.get(value, self._temporaries[expr], [offset]))

    def _load_temporary(self, expr) -> SSAValue:
        """Load the current entry of the array temporary of `expr`."""
        offset = self._temporary_offset(expr)
        return self.insert(memref.LoadOp.get(self._temporaries[expr], [offset]))

    def _needs_array(self, expr) -> bool:
        """Return whether expr is read in a different loop from the one that computes it."""
        return bool(self._temporary_indices.get(expr))

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

    def linearise_index(self, multiindex, strides) -> SSAValue:
        """ Receives a list of ints and gem.{Index, VariableIndex} and returns SSA linearised access 
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

        # FIXME: Avoiding attached cell loop
        multiindex = node.multiindex[-1]

        index_ssa = self.get_index_ssa(multiindex) 
        return self.buffer_for(variable), [index_ssa]

    def _load(self, aggregate, multiindex, strides=(1,)) -> SSAValue:
        buf = self.buffer_for(aggregate)
        return self.insert(
            memref.LoadOp.get(buf, [self.linearise_index(multiindex, strides)])
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

        # FIXME: Buffers are flat (single-indexing dim).
        # Avoiding attached cell loop 
        multiindex = multiindex[-1:]

        return self._load(aggregate, multiindex)

    # }}}

    def process(self, obj: Any, /, **kwargs):
        """Return the SSA value of `obj`, reusing a value that is already bound in scope.

        Only GEM nodes are looked up in the symbol table. A value that is
        defined inside a loop is visible only within that loop, so a later
        use outside the loop computes the value again.
        """
        if isinstance(obj, gem.Node):
            if obj in self.symbol_table:
                return self.symbol_table[obj]
            # A value that was computed in another loop is read back from its array
            if obj in self._temporaries:
                return self._load_temporary(obj)

        return self._process(obj, **kwargs)

    @functools.singledispatchmethod
    def _process(self, obj: Any, /, **kwargs):
        utils.raise_missing_dispatch_handler(obj)

    # {{{ impero types

    @_process.register(imp.Block)
    def process_block(self, tree):
        for child in tree.children:
            self.process(child)

        # An IndexSum is complete at the end of the block that initialises it. If it is
        # read in another loop, its value must be stored before the scope is left.
        if self._in_parallel_loop:
            for child in tree.children:
                if isinstance(child, imp.Initialise) and self._needs_array(child.indexsum):
                    self._store_temporary(child.indexsum, self.symbol_table[child.indexsum])

    @_process.register(imp.For)
    def process_for(self, tree):
        assert tree.index.extent

        # NOTE: Index hopefully tagged with parallel at this stage
        # if tree.index.is_parallel...
        compile_time_index_cls = (gem.Constant, numbers.Integral)

        if not isinstance(tree.index.extent, compile_time_index_cls): 
            with self.enter_parallel(tree.index):
                with self.outermost():
                    self.process(tree.children[0])

        else:
            accumulators = self.find_accumulators(tree.children[0])

            with self.enter_for(tree.index, accumulators=accumulators):
                self.process(tree.children[0])

    # NOTE: I think I can clean this up. Can I use pre_traversal instead of walk?  
    # Only got walk because of the Impero nodes. 
    def find_accumulators(self, tree) -> list[gem.IndexSum]:
        """Return the IndexSums that are accumulated within `tree` and carried as iter_args.

        MLIR is an SSA compiler, so a scalar sum that is initialised in an enclosing scope
        must be passed through the loop as an iterator argument. The arguments must be
        found before the loop is entered.

        A sum can also be initialised in a sibling loop nest. This happens when the
        summation loop is outside the loop over one of the free indices of the sum. Such a
        sum is not in scope here, and its values are stored in an array temporary. It is
        accumulated in memory by `process_accumulate`, so it is not an iter_arg.
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
        outside = [s for s in accumulated if s not in initialised]
        for s in outside:
            if s not in self.symbol_table and s not in self._temporaries:
                raise ValueError(
                    f"IndexSum {s!r} is neither bound in scope nor stored in an array"
                )
        return [s for s in outside if s in self.symbol_table]

    @_process.register(imp.Evaluate)
    def process_evaluate(self, leaf):
        """Compute the value of a GEM temporary, where Impero places it.

        The value is computed here because its operands are in scope here. For example,
        an IndexSum operand is only bound in the loop that accumulates it. The value is
        bound in the symbol table for uses in the same loop. If the temporary is also
        read in another loop, the value is stored in an array as well.

        Outside the parallel loop the computation is skipped. These temporaries do not
        depend on the parallel loop index, and each use inside the parallel loop computes
        them again on the device.
        """
        expr = leaf.expression
        # Tensor-valued constants are read through their global buffer
        if not self._in_parallel_loop or expr.shape or expr in self.symbol_table:
            return

        value = self.process(expr)
        self.symbol_table.define(expr, value)
        if self._needs_array(expr):
            self._store_temporary(expr, value)

    @_process.register(imp.Initialise)
    def process_initialise(self, leaf):
        """ Bind scalar IndexSum values to 0.

        Array-based IndexSums are already initialised. 
        """
        indexsum = leaf.indexsum
        assert not indexsum.shape

        zero = self._const(0, get_mlir_type(indexsum.dtype))
        self.symbol_table.define(indexsum, zero)

    # NOTE: I actually really hate this function. The load or scalar is so bad.
    # I think I will wait for imp.Assignment change but the design is wrong somewhere.
    @_process.register(imp.Accumulate)
    def process_accumulate(self, leaf):
        """Add a term to a scalar gem.IndexSum.

        If the sum is bound in scope, it is carried by the enclosing loops as an iter_arg.
        The sum is then rebound to the result.

        In the case of sum-factorisation, the sum may be initialised in a sibling loop.
        The result must be stored back into a temporary array. Reads are, then, also from a temp array 
        """
        indexsum = leaf.indexsum

        in_scope = indexsum in self.symbol_table
        if in_scope:
            # Inside the summation loop(s) this is the iter_arg of the innermost scf.for
            acc = self.symbol_table[indexsum]
        else:
            assert indexsum in self._temporaries, "IndexSum has not been instantiated"
            acc = self._load_temporary(indexsum)

        # The summed indices are bound by the enclosing For loops.
        term = self.process(indexsum.children[0])

        if not acc.type == term.type:
            raise TypeError(f"Accumulator and Summee should match type ({acc.type}, {term.type})")
        summed = self._binary(acc, term, arith.AddfOp, arith.AddiOp)

        if in_scope:
            self.symbol_table.update(indexsum, summed)
        else:
            self._store_temporary(indexsum, summed)

        return summed

    @_process.register(imp.Assignment)
    def process_assignment(self, leaf):
        """Store the value of leaf.expression into the buffer of leaf.assignee.

        For mode 'inc' the current value is loaded and the expression is added to it.
        If the buffer is a kernel argument and the assignment is inside the parallel
        loop, other iterations can increment the same entry concurrently. In this
        case the increment is done with a single atomic operation.
        """
        assignee = leaf.assignee
        if not isinstance(assignee, gem.Indexed):
            raise NotImplementedError(
                f"Assignment to {type(assignee).__name__} is not supported, expected gem.Indexed"
            )

        buf, idx = self._address(assignee)
        value = self.process(leaf.expression)

        if not buf.type.element_type == value.type:
            raise TypeError(f"Buffer and expression types do not match: {buf.type.element_type}, {value.type}")

        # Atomic expression if we are adding to function argument and in parallel loop
        is_atomic = assignee.children[0].name in self._arguments

        match leaf.mode:
            case "write":
                pass
            case "inc" if is_atomic:
                self.insert(
                    memref.AtomicRMWOp(
                        operands=[value, buf, idx],
                        result_types=[value.type],
                        properties={"kind": IntegerAttr(atomic_add_kind(value.type), i64)},
                    )
                )
                return
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
        """Raise an error, because the value of an IndexSum cannot be computed from its expression.

        The value is computed by `imp.Initialise` and `imp.Accumulate`. It is then read
        from the symbol table in the same loop, or from its array in another loop.
        """
        raise ValueError(
            f"IndexSum {leaf!r} is read outside the loop that computes it, and it has no array"
        )

    # }}}

    # {{{ gem operations

    # Maps each binary GEM node type to its MLIR (float_op, int_op) pair.
    # A float_op of None means that the operation is only defined for integers.
    _binary_ops = {
        gem.Sum: (arith.AddfOp, arith.AddiOp),
        gem.Product: (arith.MulfOp, arith.MuliOp),
        gem.Division: (arith.DivfOp, arith.DivSIOp),
        gem.MinValue: (arith.MinimumfOp, arith.MinSIOp),
        gem.MaxValue: (arith.MaximumfOp, arith.MaxSIOp),
        gem.FloorDiv: (None, arith.FloorDivSIOp),
        gem.Remainder: (None, arith.RemSIOp),
        gem.LogicalAnd: (None, arith.AndIOp),
        gem.LogicalOr: (None, arith.OrIOp),
    }

    @_process.register(gem.Scalar)
    def process_binary(self, expr):
        """Process a GEM node with two scalar children through the _binary_ops table.

        Other scalar nodes that do not have a more specific handler raise a missing handler error.
        """
        if type(expr) not in self._binary_ops:
            utils.raise_missing_dispatch_handler(expr)

        float_op, int_op = self._binary_ops[type(expr)]
        a, b = (self.process(c) for c in expr.children)
        return self._binary(a, b, float_op, int_op)

    # Mapping between Impero/GEM equality comparisons and MLIR for ordered/signed comparison operators
    compare_ordered_floats = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge", "==": "oeq", "!=": "one"}
    compare_signed_ints = {"<": "slt", "<=": "sle", ">": "sgt", ">=": "sge", "==": "eq", "!=": "ne"}

    @_process.register(gem.Comparison)
    def _(self, expr):
        a, b = self._align_scalars(*(self.process(c) for c in expr.children))
        if is_float(a.type):
            return self.insert(arith.CmpfOp(a, b, self.compare_ordered_floats[expr.operator]))
        return self.insert(arith.CmpiOp(a, b, self.compare_signed_ints[expr.operator]))

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
            carried through the loop as iter_args. 

        Inside the loop they are bound to the block arguments 
        After the loop they are rebound to the loop results

        This is how to sum in MLIR...
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
    def enter_parallel(self, index, is_outermost=False, accumulators=()):
        """Open an `scf.parallel` loop over index

        :arg index: numbers.Integral or gem.Index, denoting extent of domain
        :arg is_outermost: boolean flag which defines if this loop should be considered outermost loop for GPU
        :arg accumulators: keys in the symbol table (e.g. gem.IndexSum) whose values are summed over
                    
        Partial sums are completed in parallel. Need to be combined before returning

        """
        # TODO: Add GPU mapping dimensions
        assert isinstance(index, (numbers.Integral, gem.Index))

        extent = index if isinstance(index, numbers.Integral) else index.extent

        # Loop bounds must be read on host and used for setting GPU grid bounds
        self.declare_runtime_expr_on_host(extent)

        lb = self._const(0, INDEX_TYPE)
        ub = self.get_index_ssa(extent)
        step = self._const(1, INDEX_TYPE)

        init = [self.symbol_table[key] for key in accumulators]

        loop_op = scf.ParallelOp(
            [lb], [ub], [step], Region(Block(arg_types=[INDEX_TYPE])), init_vals=init
        )
        self.insert(loop_op)

        body = loop_op.body.block
        self.symbol_table.push()
        self.symbol_table.define(index, body.args[0])

        # Each iteration starts from the additive identity. The init values are added
        # only once, by the reduction.
        for key, value in zip(accumulators, init):
            self.symbol_table.define(key, self._const(0, value.type))

        self._builder_stack.append(Builder(InsertPoint.at_end(body)))

        yield

        partials = [self.symbol_table[key] for key in accumulators]

        # Need to sum each partial sum together before returning value
        regions = []
        for partial in partials:
            block = Block(arg_types=[partial.type, partial.type])
            self._builder_stack.append(Builder(InsertPoint.at_end(block)))
            summed = self._binary(*block.args, arith.AddfOp, arith.AddiOp)
            self.insert(scf.ReduceReturnOp(summed))
            self._builder_stack.pop()
            regions.append(Region(block))

        self.insert(scf.ReduceOp(partials, regions))
        self._builder_stack.pop()
        self.symbol_table.pop()

        # Include the reduced values into the scope upon exit
        for key, result in zip(accumulators, loop_op.results):
            self.symbol_table.update(key, result)

    @contextlib.contextmanager
    def outermost(self):
        """ Moves entry_depth pointer by one so that all hoisted operations go one-level deeper.

        On GPU, this is to avoid hoisted operations being moved from GPU to CPU context.
        """
        original_scopes = self.symbol_table._scopes
        original_constants = self._constants
        original_buffers = self._buffers
        original_temporaries = self._temporaries

        self.symbol_table._scopes = original_scopes[1:]
        self._constants = dict()
        self._buffers = {
            key: ssa for key, ssa in original_buffers.items() if key in self._arguments
        }
        self._temporaries = dict()
        self._entry_depth += 1

        yield

        self._entry_depth -= 1
        self.symbol_table._scopes = [original_scopes[0]] + self.symbol_table._scopes
        self._constants = original_constants
        self._buffers = original_buffers
        self._temporaries = original_temporaries


def row_major_strides(shape):
    strides = []
    acc = 1
    for extent in reversed(shape):
        strides.append(acc)
        acc *= extent
    return strides[::-1]
