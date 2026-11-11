from __future__ import annotations

import contextlib
import functools
import numbers
import types
import numpy as np
from collections.abc import Mapping
from typing import Any

import pyop3.config 
from pyop3 import utils
from pyop3.buffer import NullBuffer, ArrayBuffer, IndexedBuffer
from pyop3.device import Device, HOST_DEVICE, get_current_device
from pyop3.compile.passes.mlir.atomics import ATOMIC_RMW_KIND_ADDF, ATOMIC_RMW_KIND_ADDI
from pyop3.compile.passes.impero.utils import gem_nodes, is_shared, is_temporary, static_extent, temporary
from pyop3.compile.passes.mlir.thread_loops import THREAD_LOOP, WORKGROUP_ADDRESS_SPACE

from gem import gem, impero as imp
from gem.node import pre_traversal

from xdsl.dialects import arith, func, gpu, memref, scf
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
    NoneAttr,
    StringAttr,
    TensorType,
    UnitAttr,
)

INDEX_TYPE = IndexType()
FLOAT_TYPES = (f32, f64)

# ffast-math flag. No MLIR pass can apply this so has to be done at codegen.
FP_FAST = arith.FastMathFlagsAttr("fast")

# LLVM combines adjacent shared-memory loads into 16-byte vector loads, so each
# shared temporary must start on a 16-byte boundary.
SHARED_ALIGNMENT = 16

def tables(node: imp.Node) -> list[gem.Literal]:
    """The tensor-valued literals below `node`, without repetition."""
    return list(dict.fromkeys(n for n in gem_nodes(node) if isinstance(n, gem.Literal) and n.shape))


def shared_table_bytes(node: imp.Node) -> int:
    """The shared memory that the tables below `node` take when a builder has `shared_tables`."""
    return sum(utils.ceildiv(t.array.nbytes, SHARED_ALIGNMENT) * SHARED_ALIGNMENT for t in tables(node))


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
        shared_tables: bool = False,
    ) -> None:
        """Create the kernel FuncOp, with one memref argument for each buffer in `arguments`.

        :arg root: Impero tree that `build` translates. A value that one loop computes
            and another loop reads must already be stored in a temporary, see
            `pyop3.compile.passes.impero.item_loop.assign_crossing_values`.
        :arg arguments: Map from the kernel name of each buffer to its buffer view.
            Temporaries (views of a `NullBuffer`) are not function arguments and are dropped.
        :arg name: The name of the generated MLIR kernel
        :arg shared_tables: Whether each block of an `ItemBatch` copies the tables into its shared memory.
            The caller checks that they fit, see `shared_table_bytes`.

        """
        self._root = root
        self._shared_tables = shared_tables

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

        # Kernel name -> SSAValue of the function argument
        self._argument_values: dict[str, SSAValue] = dict(zip(self._arguments, entry_block.args))

        # The block that values are hoisted to: the function body, or the body of the item loop
        # while it is generated, so that each item (each GPU thread) gets its own temporaries
        self._entry: Block = entry_block

        # (Hoisting block, variable name or gem.Literal) -> SSAValue of the buffer declared in the block
        self._buffers: dict[tuple[Block, Any], SSAValue] = {}

        # Constant attribute -> SSAValue of its arith.constant in the function body
        self._constants: dict[FloatAttr | IntegerAttr, SSAValue] = {}

        self.symbol_table = SymbolTable()

        # Globally defined arrays - like quadrature weights
        self.global_ops: list[Operation] = list()

        self._name_generator = utils.UniqueNameGenerator()

        # Argument name -> Device that the argument must be placed on
        self._argument_devices: dict[str, Device] = dict.fromkeys(self._arguments, get_current_device())


    @property
    def builder(self) -> Builder:
        return self._builder_stack[-1]

    @property
    def _in_item_loop(self) -> bool:
        """Whether code is currently generated inside the loop over the items."""
        return self._entry is not self._func_op.body.block

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

        return str(modop), self.arguments, f"_mlir_ciface_{self._func_op.sym_name.data}"

    # {{{ helpers

    def unique_name(self, prefix: str) -> str:
        return self._name_generator(prefix)

    def insert(self, op: Operation) -> SSAValue | None:
        """Insert an MLIR operation into the current block, returning its first result."""
        self.builder.insert(op)
        results = op.results
        return results[0] if results else None

    def insert_at_entry(self, op: Operation) -> SSAValue:
        """Insert an MLIR operation at the start of the hoisting block, so that it dominates every use."""
        Builder(InsertPoint.at_start(self._entry)).insert(op)
        return op.results[0]

    def _const(self, value, mlir_type) -> SSAValue:
        """Return the SSA value of a scalar constant, reusing it if it already exists.

        Constants are defined at the start of the function body, so that they dominate
        every use. `gpu-kernel-outlining` copies the constants that a GPU kernel uses into it.
        """
        if is_float(mlir_type):
            attr = FloatAttr(float(value), mlir_type)
        else:
            attr = IntegerAttr(int(value), mlir_type)
        if attr not in self._constants:
            op = Builder(InsertPoint.at_start(self._func_op.body.block)).insert(arith.ConstantOp(attr))
            self._constants[attr] = op.result
        return self._constants[attr]

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

    def _to_float(self, ssa: SSAValue) -> SSAValue:
        """Convert an integer scalar to f64, because the MLIR math functions take floating-point operands."""
        return ssa if is_float(ssa.type) else self._convert_type(ssa, f64)

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
        Floating-point operations carry the `fast` fast-math flags, so that LLVM can, for example,
        fuse a multiply and the add that uses its result into one FMA instruction.
        """
        a, b = self._align_scalars(a, b)
        if is_float(a.type):
            if float_op is None:
                raise TypeError(f"{int_op.__name__} is not defined for floating point operands")
            return self.insert(float_op(a, b, flags=FP_FAST))
        return self.insert(int_op(a, b))

    # }}}

    # {{{ buffers

    def buffer_for(self, node) -> SSAValue:
        """Return the SSA value of the buffer that backs `node`, declaring it if needed.

        A `gem.Variable` becomes either a temporary allocation or a function
        argument, and a `gem.Literal` becomes a read-only global constant.
        """
        if isinstance(node, gem.Variable) and node.name in self._argument_values:
            return self._argument_values[node.name]
        key = (self._entry, node.name if isinstance(node, gem.Variable) else node)
        if key not in self._buffers:
            self._buffers[key] = self._declare_buffer(node)
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
        iteration (each GPU thread) gets its own copy of the temporary, in the
        private address space of the device. The allocation is not initialised,
        so each entry must be written before it is read.

        A shared temporary is a `memref.global` in the workgroup address space,
        which `gpu-kernel-outlining` copies into the GPU module.
        """
        # Place node in shared memory if possible 
        if is_shared(node):
            # Memory space assigns location on the GPU (i.e. L1 shared memory vs global memory (per-batch vs per-kernel))
            memref_type = MemRefType(get_mlir_type(node.dtype), node.shape, memory_space=WORKGROUP_ADDRESS_SPACE)
            name = self.unique_name(f"shared_{node.name}")
            self.global_ops.append(memref.GlobalOp.get(
                StringAttr(name), memref_type, UnitAttr(), sym_visibility=StringAttr("private"), alignment=SHARED_ALIGNMENT,
            ))

            return self.insert_at_entry(memref.GetGlobalOp(name, memref_type))

        if not is_temporary(node):
            raise ValueError(f"{node.name!r} is neither a temporary nor a kernel argument")

        # Non-shared threads should be private/per-thread temporaries on GPU 
        space = get_current_device().private_address_space if self._in_item_loop else None
        memory_space = NoneAttr() if space is None else IntegerAttr(space, i64)
        return self.insert_at_entry(memref.AllocaOp.get(get_mlir_type(node.dtype), shape=node.shape, memory_space=memory_space))


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

        index, = node.multiindex
        index_ssa = self.get_index_ssa(index)
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
        strides = utils.row_major_strides([mi.extent for mi in multiindex])
        return self._load(aggregate, multiindex, strides)

    @index_into.register(gem.Variable)
    def _(self, aggregate, multiindex) -> SSAValue:
        return self._load(aggregate, multiindex)

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

        # Only the outermost parallel loop is mapped to `scf.parallel`. A parallel loop
        # inside it, for example the loop over the layers of an extruded mesh, runs
        # sequentially in each iteration.
        if tree.is_parallel and not self._in_item_loop:
            with self.enter_parallel(tree.index):
                self.process(tree.children[0])

        else:
            accumulators = self.find_accumulators(tree.children[0])

            with self.enter_for(tree.index, accumulators=accumulators):
                self.process(tree.children[0])

    @_process.register(imp.ItemBatch)
    def process_item_batch(self, batch):
        """Launch one block of `batch.block_size` threads per batch of items.

        At least one block is launched, so that the grid is valid when there are no items.
        Each ThreadLoop of the batch runs over the items of the
        batch and its domain, and skips the items past the end.
        """

        extent = batch.index.extent
        self.declare_runtime_expr_on_host(extent)
        num_items = self.get_index_ssa(extent)
        one = self._const(1, INDEX_TYPE)

        num_blocks = self.insert(arith.CeilDivUIOp(num_items, self._const(batch.batch_size, INDEX_TYPE)))
        num_blocks = self.insert(arith.MaxUIOp(num_blocks, one))
        block_size = self._const(batch.block_size, INDEX_TYPE)

        # The arguments are the block ids, thread ids, grid sizes and block sizes, each (x, y, z)
        body = Block(arg_types=[INDEX_TYPE] * 12)
        self.insert(gpu.LaunchOp(Region(body), [num_blocks, one, one], [block_size, one, one]))

        block_id = body.args[0]
        with self._item_loop(body):
            # One thread per entry copies each table, so that the phases read it from shared memory
            for table in tables(batch) if self._shared_tables else ():
                source = self._declare_buffer(table)
                shared = self.buffer_for(temporary(self.unique_name("table"), (table.array.size,), table.dtype, shared=True))
                with self.thread_loop([table.array.size]) as (entry,):
                    self.insert(memref.StoreOp.get(self.insert(memref.LoadOp.get(source, [entry])), shared, [entry]))
                self._buffers[self._entry, table] = shared

            # Each child of ItemBatch is a ThreadLoop. Each loop does not iterate but assigns steps to threads
            for loop in batch.statement.children:
                with self.thread_loop([batch.batch_size, *map(static_extent, loop.domain)]) as (local, *indices):

                    first = self.insert(arith.MuliOp(block_id, self._const(batch.batch_size, INDEX_TYPE)))
                    item = self.insert(arith.AddiOp(first, local))

                    # Making sure that only threads less than extent do compute 
                    with self._if(self.insert(arith.CmpiOp(item, num_items, "ult"))):
                        # Defining cell and local thread index 
                        self.symbol_table.define(batch.index, item)
                        self.symbol_table.define(batch.local_index, local)

                        for index, value in zip(loop.domain, indices, strict=True):
                            self.symbol_table.define(index, value)
                        
                        self.process(loop.statement)

            self.insert(gpu.TerminatorOp())

    def find_accumulators(self, tree) -> list[gem.IndexSum]:
        """Return the IndexSums that are accumulated within `tree` and carried as iter_args.

        MLIR is an SSA compiler, so a scalar sum that is initialised in an enclosing scope
        must be passed through the loop as an iterator argument. The arguments must be
        found before the loop is entered.
        """
        nodes = list(pre_traversal([tree]))
        accumulated = dict.fromkeys(n.indexsum for n in nodes if isinstance(n, imp.Accumulate))
        initialised = {n.indexsum for n in nodes if isinstance(n, imp.Initialise)}
        outside = [s for s in accumulated if s not in initialised]
        if missing := [s for s in outside if s not in self.symbol_table]:
            raise ValueError(f"IndexSums {missing!r} are accumulated outside the loop that initialises them")
        return outside

    @_process.register(imp.Evaluate)
    def process_evaluate(self, leaf):
        """Compute the value of a GEM temporary, where Impero places it.

        The value is computed here because its operands are in scope here. For example,
        an IndexSum operand is only bound in the loop that accumulates it. The value is
        bound in the symbol table for uses in the same loop.

        Outside the parallel loop the computation is skipped. These temporaries do not
        depend on the parallel loop index, and each use inside the parallel loop computes
        them again on the device.
        """
        expr = leaf.expression
        # Tensor-valued constants are read through their global buffer
        if not self._in_item_loop or expr.shape or expr in self.symbol_table:
            return
        self.symbol_table.define(expr, self.process(expr))

    @_process.register(imp.Initialise)
    def process_initialise(self, leaf):
        """Bind a scalar IndexSum to 0."""
        indexsum = leaf.indexsum
        assert not indexsum.shape

        zero = self._const(0, get_mlir_type(indexsum.dtype))
        self.symbol_table.define(indexsum, zero)

    @_process.register(imp.Accumulate)
    def process_accumulate(self, leaf):
        """Add a term to a scalar gem.IndexSum, and rebind the sum to the result

        Inside the summation loops the sum is the iter_arg of the innermost scf.for
        """
        indexsum = leaf.indexsum
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
        """Store the value of leaf.expression into the buffer of leaf.assignee

        An `inc` loads, adds and stores. An `atomic` is an increment that other
        threads may do at the same time, see `mark_atomics` to see how atomics flag attached
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

        match leaf.mode:
            case "write":
                pass
            case "atomic":
                self.insert(
                    memref.AtomicRMWOp(
                        operands=[value, buf, idx],
                        result_types=[value.type],
                        properties={"kind": IntegerAttr(
                            ATOMIC_RMW_KIND_ADDF if is_float(value.type) else ATOMIC_RMW_KIND_ADDI, i64
                        )},
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

        The value is computed by `imp.Initialise` and `imp.Accumulate`, and read from the
        symbol table in the same loop. A sum that another loop reads is stored in a
        temporary by the "assign_crossing_values" compiler pass
        """
        raise ValueError(f"IndexSum {leaf!r} is read outside the loop that computes it")

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
        cond, then, else_ = (self.process(c) for c in expr.children)
        then, else_ = self._align_scalars(then, else_)
        return self.insert(arith.SelectOp(cond, then, else_))

    # Maps each GEM math function name to the MLIR op that computes it on floats.
    _math_ops = {
        "abs": xmath.AbsFOp,
        "sqrt": xmath.SqrtOp,
        "exp": xmath.ExpOp,
        "ln": xmath.LogOp,
        "sin": xmath.SinOp,
        "cos": xmath.CosOp,
        "tan": xmath.TanOp,
        "sinh": xmath.SinhOp,
        "cosh": xmath.CoshOp,
        "tanh": xmath.TanhOp,
        "asin": xmath.AsinOp,
        "acos": xmath.AcosOp,
        "atan": xmath.AtanOp,
        "atan2": xmath.Atan2Op,
        "erf": xmath.ErfOp,
    }

    @_process.register(gem.MathFunction)
    def _(self, expr):
        args = [self.process(c) for c in expr.children]
        if expr.name == "abs" and not is_float(args[0].type):
            return self.insert(xmath.AbsIOp(args[0]))
        if expr.name not in self._math_ops:
            raise NotImplementedError(f"Math function {expr.name!r} is not implemented")
        args = [self._to_float(a) for a in args]
        if len(args) == 2:
            args = self._align_scalars(*args)
        return self.insert(self._math_ops[expr.name](*args))

    @_process.register(gem.Power)
    def _(self, expr):
        """Compute a power in floating point, as the C `pow` of the loopy backend does.

        The MLIR lowering has no pass for `math.ipowi`, so integer powers are also computed in floating point.
        """
        base, exponent = (self._to_float(self.process(c)) for c in expr.children)
        return self.insert(xmath.PowFOp(*self._align_scalars(base, exponent)))

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
        with self._scope(body):
            self.symbol_table.define(index, body.args[0])

            for key, arg in zip(accumulators, body.args[1:]):
                self.symbol_table.define(key, arg)
            
            yield
            self.insert(scf.YieldOp(*(self.symbol_table[key] for key in accumulators)))

        # Include the summed arguments into the scope upon exit  
        for key, result in zip(accumulators, loop_op.results):
            self.symbol_table.update(key, result)

    @contextlib.contextmanager
    def enter_parallel(self, index: gem.Index):
        """Open an `scf.parallel` loop over `index`, whose extent is read on the host to size the GPU grid."""
        self.declare_runtime_expr_on_host(index.extent)

        lb = self._const(0, INDEX_TYPE)
        ub = self.get_index_ssa(index.extent)
        step = self._const(1, INDEX_TYPE)
        body = Block(arg_types=[INDEX_TYPE])

        self.insert(scf.ParallelOp([lb], [ub], [step], Region(body)))

        with self._item_loop(body):
            self.symbol_table.define(index, body.args[0])
            yield
            self.insert(scf.ReduceOp())

    @contextlib.contextmanager
    def thread_loop(self, extents: list[int]):
        """Open an `scf.parallel` over `extents`, marked for the passes in
        `pyop3.compile.passes.mlir.thread_loops`, and yield its indices."""
        zero, one = self._const(0, INDEX_TYPE), self._const(1, INDEX_TYPE)
        body = Block(arg_types=[INDEX_TYPE] * len(extents))

        loop_op = scf.ParallelOp(
            [zero] * len(extents), [self._const(e, INDEX_TYPE) for e in extents], [one] * len(extents), Region(body)
        )
        loop_op.attributes[THREAD_LOOP] = UnitAttr()
        self.insert(loop_op)

        with self._scope(body):
            yield body.args
            self.insert(scf.ReduceOp())

    @contextlib.contextmanager
    def _if(self, condition: SSAValue):
        """Open an `scf.if` without results or else branch."""
        then = Block()
        self.insert(scf.IfOp(condition, [], Region(then)))
        with self._scope(then):
            yield
            self.insert(scf.YieldOp())

    @contextlib.contextmanager
    def _scope(self, block: Block):
        """Generate code at the end of `block`, in a new symbol-table scope."""
        self.symbol_table.push()
        self._builder_stack.append(Builder(InsertPoint.at_end(block)))

        yield
        
        self._builder_stack.pop()
        self.symbol_table.pop()

    @contextlib.contextmanager
    def _item_loop(self, block: Block):
        """Generate the loop over the items at the end of `block`, hoisting values to its start."""
        assert not self._in_item_loop, "Item loops do not nest"
        self._entry = block
        with self._scope(block):
            yield
        self._entry = self._func_op.body.block
