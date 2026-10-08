from __future__ import annotations

import functools
import contextlib
import dataclasses
import numbers
import types
import numpy as np
from typing import Any

from immutabledict import immutabledict as idict

from pyop3 import utils
from pyop3.buffer import NullBuffer, ArrayBuffer

from gem import gem, impero as imp 

from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects import math as xmath
from xdsl.builder import Builder, InsertPoint
from xdsl.ir import SSAValue, Block, Region, Operation

from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    DictionaryAttr,
    DYNAMIC_INDEX,
    IntegerType,
    IndexType,
    IntegerAttr,
    Float64Type,
    FloatAttr,
    i32, i64, f32, f64,
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
    shape: Tuple[int] | None

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

class SymbolTable:
    """
    Class deals with SSAValues for a respective scope.

    e.g: SSAValues inside scf.ForOp are not valid outside the iterative loop.
    """
    def __init__(self) -> None:
      self._scopes: List[Dict[Any, SSAValue]] = [dict()]

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


class MLIRBuilder():
    """Visitor that populates an MLIR Object.

    This class is intended to be subclassed by 'reconstruction' visitors that
    build similar objects.

    """
    def __init__(self) -> None:
      # If we can know function arguments in advance, we can create FuncOp as opposed to Block
      self._entry_block = Block()

      # Builder stack - Builder object acts as bridge to generate code for a given reg/block/op
      self._builder_stack: List[Builder] = [
          Builder(InsertPoint.at_end(self._entry_block))
      ] 

      # Symbol table (buffers in outer scope, inames/loop-idx in inner)
      self.symbol_table = SymbolTable()

      # buffer identity -> block-arg SSAValue
      self._buffer_args: Dict[Any, SSAValue] = dict()

      # iname -> (start, stop) domain window
      self._domains: Dict[str, Tuple[SSAValueT, SSAValueT]] = dict()

      # GEM Node -> (memref, free_indices); storing temporary variables (i.e. t1, t2) 
      self._temporaries: dict[Any, tuple[SSAValue, tuple]] = dict()

      # Going to need this for globally defined constants - like quadrature weights 
      self.global_ops: list[Operations] = list()
      self.globals: dict[Any, SSAValue] = dict() 
      self.literal_to_name: dict[gem.Literal, str] = dict()

      self._name_generator = utils.UniqueNameGenerator()

      # Arguments to return to buffer
      self._arguments: list[Argument] = list()


    @property
    def builder(self) -> Builder:
      return self._builder_stack[-1]

    @property
    def arguments(self) -> list: 
      return sorted(self._arguments, key=lambda arg: arg.name)

    def build(self, root, name: str = "imp_kernel", args=()) -> ModuleOp:
      """ 
      Construct the MLIR module, containing a FuncOp, given an Impero tree.

      :arg root: ImperoC tree
      :arg name: Name of generated MLIR kernel
      :arg args: gem.Variable in desired argument order. gem.Variables not provided are appended in order of visitation
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

    def insert(self, op: Operation) -> SSAValue:
      """ Inserts an MLIR operation into the block, returning SSA result """
      self.builder.insert(op)
      results = op.results 
      return results[0] if results else None 
  
    def _const(self, value, mlir_type) -> SSAValue:
      if is_float(mlir_type):
          attr = FloatAttr(float(value), mlir_type)
      else:
          attr = IntegerAttr(int(value), mlir_type)
      return self.insert(arith.ConstantOp(attr))

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
      if isinstance(ssa.type, IndexType):
        return ssa
      return self.insert(arith.IndexCastOp(ssa, INDEX_TYPE))

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
      a, b = self._align_scalars(a, b)
      return self.insert((float_op if is_float(a.type) else int_op)(a, b))

    def get_extent(self, index) -> int:
      """ 

      Return extent for index as SSA 

      """
      extent = index.extent

      if isinstance(extent, gem.Literal):
        return int(extent.value)
      elif isinstance(extent, numbers.Integral):
        return self.get_index_ssa(extent)
      elif isinstance(extent, gem.Index):
        return  
      else:
        raise NotImplementedError(f"Not prepared for value of type: {extent!r}")

    # }}} 

    # {{{ buffers and temporaries
  
    @functools.singledispatchmethod
    def collect_buffer(self, node) -> SSAValue:
      """ Returns SSA value corresponding to buffer 

      Buffer exists in either:
      - self.globals as constant read-only, or
      - self._buffer_args as a function/temporary argument 

      If it does not exist, the buffer is registered
      """
      utils.raise_missing_dispatch_handler(node)

    @collect_buffer.register(gem.Literal)
    def _(self, literal) -> SSAValue:
      # This confusing indirection is because I have not added a name to the gem.Literal object
      if literal not in self.literal_to_name:
        self.register_buffer(literal)

      name = self.literal_to_name[literal]
      return self.globals[name]
        

    @collect_buffer.register(gem.Variable)
    def _(self, variable) -> SSAValue:
      if variable.name not in self._buffer_args:
        self.register_buffer(variable) 

      return self._buffer_args[variable.name] 
    
    @functools.singledispatchmethod
    def register_buffer(self, node) -> SSAValue:
      utils.raise_missing_dispatch_handler(node)

    @register_buffer.register(gem.Variable)
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
      
      # Working with temporary 
      if isinstance(node.data.buffer, NullBuffer):
        """
        NOTE: AllocOps are used because AllocaOp bad on GPU
        Compiler pass can raise Alloc to Alloca
        Alloca = stack allocate, Alloc = heap allocate
        Deallocations required for alloc (and inserted with compiler pass)
        """
        op = memref.AllocOp.get(mlir_type, shape=shape)
        
        # Insert temporary tensor at highest level (outside current brace nesting) 
        Builder(InsertPoint.at_start(self._entry_block)).insert(op)
        self._buffer_args[node.name] = op.results[0]
        
        # Allocate to 0 values, iterating over allocated shape  
        # The positioning of this is very wrong. 
        # I can come back to this. Focus on segfault now
        zero = self._const(0, mlir_type)
        extent = shape[0]
        with self.enter_for(extent):
          i = self.symbol_table[extent]
          self.insert(memref.StoreOp.get(zero, op.results[0], [i]))


      # Working with function argument
      # TODO: Maybe want to assign as dynamic indices here...
      elif isinstance(node.data.buffer, ArrayBuffer):
        memref_type = MemRefType(mlir_type, shape=[DYNAMIC_INDEX])
        arg = self._entry_block.insert_arg(memref_type, len(self._entry_block.args)) 
        arg.name_hint = node.name 
        self._buffer_args[node.name] = arg 

        self._arguments.append(Argument(node.name, node.dtype, shape))
          
        # At this point, we need to add to buffer
      
      return self._buffer_args[node.name]

    @register_buffer.register(gem.Literal)
    def _(self, node: gem.Literal) -> SSAValue:
      """ Define Literal (tensor-valued constant) as global constant read-only buffer """ 
      if node in self.globals:
        return self.globals[node] 
      
      arr = node.array 
      mlir_type = get_mlir_type(node.dtype)
      memref_type = MemRefType(mlir_type, arr.shape)
      # FIXME: Likely that the unique name portion of this is not working correctly 
      name = self.unique_name("literal")
      self.literal_to_name[node] = name 
      
      value = DenseIntOrFPElementsAttr.from_list(
        TensorType(mlir_type, arr.shape), arr.data
      )
      
      # Registering the constant array 
      global_op = memref.GlobalOp.get(
        StringAttr(name),
        memref_type,
        value,
        sym_visibility=StringAttr("private"),
        constant=UnitAttr()
      )
      self.global_ops.append(global_op)

      get_ssa = self.insert(memref.GetGlobalOp(name, memref_type))

      self.globals[name] = get_ssa

      return self.globals[name]
    
    def _temp_indices(self, key) -> list[SSAValue]:
      _, free = self._temporaries[key]
      return [self.symbol_table[i] for i in free]

    @functools.singledispatchmethod
    def get_index_ssa(self, idx) -> SSAValue:
      utils.raise_missing_dispatch_handler(idx)

    @get_index_ssa.register(numbers.Integral)
    def _(self, idx: numbers.Integral):
      return self._const(idx, INDEX_TYPE)

    @get_index_ssa.register(gem.Constant)
    def _(self, idx: gem.Constant): 
      return self._const(idx.value, INDEX_TYPE)

    @get_index_ssa.register(IndexType)
    def _(self, idx: INDEX_TYPE):
      return self.symbol_table[idx]
      
    @get_index_ssa.register(gem.VariableIndex)
    def _(self, idx: gem.VariableIndex):
      emitted = self.emit(idx.expression)
      res = self._to_index(emitted)
      
      return res 

    @get_index_ssa.register(gem.Index)
    def _(self, idx: gem.Index):
      if idx in self.symbol_table:
        return self.symbol_table[idx]
      
      rhs = self.process(idx.extent)
      return self._to_index(rhs)  

    @get_index_ssa.register(gem.Node)
    def _(self, node: gem.Node):
      ssa = self.process(node)
      return self._to_index(ssa)

    @functools.singledispatchmethod
    def get_address(self, node: Any):
      """(memref, indices) for an lvalue: Variable, or Indexed/Gather of one."""
      utils.raise_missing_dispatch_handler(node)

    @get_address.register(gem.Indexed)
    @get_address.register(gem.Gather)
    def _(self, node): 
      # TODO: Figure out why this assert is actually true
      assert isinstance(node.children[0], gem.Variable)
      buffer, strided_index = (self.collect_buffer(node.children[0]), self.get_strided_index(node.children[0], node.multiindex)) 
      return (buffer, strided_index) 

    # NOTE: Might avoid a dispatch function at this point 
    # TODO: Get rid of this function. Overlaps with 'process'
    def emit(self, expr) -> SSAValue:
      """Value of a GEM expression, reusing an Evaluate/IndexSum temporary if one exists."""
      if isinstance(expr, numbers.Number):
        return self._const(expr, i64 if isinstance(expr, numbers.Integral) else f64)
      if isinstance(expr, gem.Node) and expr in self._temporaries:
        buf, _ = self._temporaries[expr]
        return self.insert(memref.LoadOp.get(buf, self._temp_indices(expr)))
      return self.process(expr)


    def linearise_index(self, strides, multiindex) -> SSAValue:
      """ Receives a list of ints and gem.{Index, VariableIndex} and returns SSA linearised access """
      linearised = None

      # linearised = sum(index * stride)
      for ind, step in zip(multiindex, strides):
        term = self._binary(
          self.get_index_ssa(ind), self._const(int(step), INDEX_TYPE),
          arith.MulfOp, arith.MuliOp
        )

        linearised = term if linearised is None else self._binary(linearised, term, arith.AddfOp, arith.AddiOp)
      return linearised 

#       # Simulating a ternary operator in MLIR
#       # Nested Select(Select(Select(...)))
#       # Each element expression is eagerly evaluated this way 
#       acc = self.emit(arr.flat[-1])
#       for k in range(arr.size-2, -1, -1):
#         hit = self.insert(arith.CmpiIOp(linearised, self._const(k, INDEX_TYPE), "eq"))
#         val = self.emit(arr.flat[k])
#         val, acc = self._align_scalars(val, acc)
            
#         acc = self.insert(arith.SelectOp(hit, val, acc))
#       return acc

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
      eq = self.insert(arith.CmpiIOp(i, j, "eq"))
      t = get_mlir_type(aggregate.dtype)
      return self.insert(arith.SelectOp(eq, self._const(1, t), self._const(0, t)))

    @index_into.register(gem.Literal)
    def _(self, aggregate, multiindex) -> SSAValue:
      buf = self.collect_buffer(aggregate)

      strides = [1]

      linearised = [self.linearise_index(strides, multiindex)]

      return self.insert(
        memref.LoadOp.get(
          buf, linearised
        )
      )

    @index_into.register(gem.Variable)
    def _(self, aggregate, multiindex) -> SSAValue:
      buf = self.collect_buffer(aggregate)

      # TODO: Maybe find a way to refactor this in future 
      variable, dim2idxs, indexes = gem.decompose_variable_view(aggregate)
      strides = [stride for _, idxs in dim2idxs for _, stride in idxs] 

      linearised = [self.linearise_index(strides, multiindex)]

      return self.insert(
        memref.LoadOp.get(
          buf, linearised
        )
      )

    @index_into.register(gem.ListTensor)
    def _(self, aggregate: gem.ListTensor, multiindex) -> SSAValue:
      """ Indexing into ListTensor (stack or list of tensors essentially) 

      I think there is some optimisations to make here regarding the loading of the required elements.
      All elements are evaluated here, regardless of indexing, which is not ideal
      """
      
      arr = aggregate.array

      # If multiindex all compile-time integers, return the indices
      if all(isinstance(i, numbers.Integral) for i in multiindex):
        return self.emit(arr[tuple(int(i) for i in multiindex)])

      # Else, parse indices for runtime-indices of tensor and accumulate
      strides = np.cumprod((1,) + arr.shape[:0:-1])[::-1]
      linearised = None

      # linearised = sum(index * stride)
      for ind, step in zip(multiindex, strides):
        term = self._binary(
          self.get_index_ssa(ind), self._const(int(step), INDEX_TYPE),
          arith.MulfOp, arith.MuliOp
        )

        linearised = term if linearised is None else self._binary(linearised, term, arith.AddfOp, arith.AddiOp)

      # Simulating a ternary operator in MLIR
      # Nested Select(Select(Select(...)))
      # Each element expression is eagerly evaluated this way 
      acc = self.emit(arr.flat[-1])
      for k in range(arr.size-2, -1, -1):
        hit = self.insert(arith.CmpiIOp(linearised, self._const(k, INDEX_TYPE), "eq"))
        val = self.emit(arr.flat[k])
        val, acc = self._align_scalars(val, acc)
            
        acc = self.insert(arith.SelectOp(hit, val, acc))
      return acc

    @index_into.register(gem.ComponentTensor)
    def _(self, aggregate, multiindex) -> SSAValue:
      values = [self.get_index_ssa(i) for i in multiindex]

      # New symbol table so that operations within ComponentTensor use local SSA  
      self.symbol_table.push()
      try:
        for j, v in zip(aggregate.multiindex, values):
          self.symbol_table.define(j, v)
        return self.emit(aggregate.children[0])
      finally:
        self.symbol_table.pop()
    
    # }}} 

    def get_strided_index(self, aggregate, multiindex) -> SSAValue: 
      variable, dim2idxs, indexes = gem.decompose_variable_view(aggregate)
      strides = [stride for _, idxs in dim2idxs for _, stride in idxs] 

      linearised = [self.linearise_index(strides, multiindex)]
      return linearised

    @functools.singledispatchmethod
    def process(self, obj: Any, /, **kwargs):
      utils.raise_missing_dispatch_handler(obj)

    # {{{ impero types

    @process.register(imp.Block)
    def process_block(self, tree): 
      # NOTE: Don't think Blocks are needed in MLIR relative to IR 
      for child in tree.children:
        self.process(child)

    @process.register(imp.For)
    def process_for(self, tree): 
      assert tree.index.extent

      # NOTE: Index hopefully tagged with parallel at this stage  
      # if tree.index.is_parallel... 

      with self.enter_for(tree.index):
        self.process(tree.children[0])

    # NOTE: This is not being used. Not sure why. 
    @process.register(imp.Initialise)
    def process_initialise(self, leaf): 
      raise NotImplementedError
      # isum = leaf.indexsum 
      # dtype = get_mlir_type(isum.dtype)
      # buf = self._alloc_temp(isum, isum.free_indices, dtype)
      # sop = memref.StoreOp.get(self._const(0, dtype), buf, self._temp_indices(isum))
      # self.insert(sop)

    @process.register(imp.Accumulate)
    def process_accumulate(self, leaf): 
      """ Process gem.IndexSum  """ 
      isum = leaf.indexsum
      if isum not in self._temporaries:
          raise KeyError(f"Accumulate before Initialise for {isum}")
      buf, _ = self._temporaries[isum]
      idx = self._temp_indices(isum)
      term = self._convert_type(self.emit(isum.children[0]), buf.type.element_type)
      old = self.insert(memref.LoadOp.get(buf, idx))
      new = self._binary(old, term, arith.AddfOp, arith.AddiOp)
      sop = memref.StoreOp.get(new, buf, idx) 
      
      self.insert(sop)

    @process.register(imp.Return)
    def process_return(self, leaf):
      """ Store value of expression into variable 
      
      If variable is Indexed - we store into the buffer
      If variable is not Indexed, we just store by name into symbol table
    
      """ 
        
      var = leaf.variable 
      
      # If variable is just a temporary scalar
      if not isinstance(var, gem.Indexed):
        rhs = self.process(var.expression)
        self.symbol_table[var.name] = rhs
        return

      buf, idx = self.get_address(var) 
      value = self.process(leaf.expression)

      sop = memref.StoreOp.get(value, buf, idx)
      self.insert(sop)
      return 

    # FIXME: Bug is here. 
    @process.register(imp.Evaluate)
    def process_evaluate(self, leaf):
      """ Calculate value within expression and assign to temporary """
      expr = leaf.expression
      
      # if "'t_2" in repr(expr):
      #   breakpoint()
      
      if expr in self.symbol_table:
        return self.symbol_table[expr]

      value = self.process(expr)
      self.symbol_table.define(expr, value)
      return self.symbol_table[expr]

    # }}}

    # {{{ gem types 
    
    @process.register(gem.Zero)
    def _(self, leaf):
      assert not leaf.shape
      return self._const(0, get_mlir_type(leaf.dtype))
    
    @process.register(gem.Literal)
    def _(self, leaf):
      """ Process gem.Literal which is a tensor-valued constant """
      # Check if GEM is a tensor-valued expression
      if leaf.shape:
        return 
      else:    
        return self._const(leaf.value, get_mlir_type(leaf.dtype))

    # FIXME: This implementation makes no sense 
    @process.register(gem.Variable)
    def _(self, leaf):
      raise NotImplementedError
      if leaf.shape:
        raise NotImplementedError(f"Variable {leaf.name} is tensor-valued; it must be indexed")
      return self.insert(memref.LoadOp.get(self.collect_buffer(leaf), []))

    @process.register(gem.Index)
    def _(self, leaf):
      if leaf not in self.symbol_table:
        # Really hope that this exists for all indices
        assert leaf.extent.value 
        
        # TODO: Should be using get_index_ssa
        index_ssa = self._const(leaf.extent.value, INDEX_TYPE) 
        self.symbol_table.define(leaf, index_ssa)

      return self.symbol_table[leaf] 

    # TODO: I am not parsing this correctly 
    @process.register(gem.VariableIndex)
    def _(self, leaf):
      return self.emit(leaf.expression)

    @process.register(gem.ListTensor)
    def _(self, leaf):
      raise NotImplementedError("ListTensor is only valid beneath Indexed")
      
    @process.register(gem.ComponentTensor)
    def _(self, leaf):
      raise NotImplementedError("ComponentTensor is only valid beneath Indexed")

    @process.register(gem.IndexSum)
    def _(self, leaf):
      raise KeyError(f"IndexSum should be nested inside imp.Initialise/Accumulate...")

    @process.register(gem.Indexed)
    @process.register(gem.Gather)
    def _(self, leaf):
      return self.index_into(leaf.children[0], leaf.multiindex) 

    # }}}

    # {{{ gem operations 

    @process.register(gem.Sum)
    def _expression_sum(self, expr):
      a, b = (self.process(c) for c in expr.children)
      return self._binary(a, b, arith.AddfOp, arith.AddiOp)

    @process.register(gem.Product)
    def _expression_product(self, expr):
      a, b = (self.process(c) for c in expr.children)
      return self._binary(a, b, arith.MulfOp, arith.MuliOp)

    @process.register(gem.Division)
    def _expression_division(self, expr):
      a, b = (self.process(c) for c in expr.children)
      return self._binary(a, b, arith.DivfOp, arith.DivSIOp)

    @process.register(gem.FloorDiv)
    def _expression_floordiv(self, expr):
      a, b = (self.process(c) for c in expr.children)
      a, b = self._align_scalars(a, b)

      assert not is_float(a.type)

      return self.insert(arith.FloorDivSIOp(a, b))

    @process.register(gem.Remainder)
    def _expression_remainder(self, expr):
      a, b = (self.process(c) for c in expr.children)
      a, b = self._align_scalars(a, b)
      
      assert not is_float(a.type)
       
      return self.insert(arith.RemSIOp(a, b))

    @process.register(gem.LogicalAnd)
    def _expression_logicaland(self, expr):
      a, b = (self.process(c) for c in expr.children)
      return self.insert(arith.AndIOp(a, b))

    @process.register(gem.LogicalOr)
    def _expression_logicalor(self, expr):
      a, b = (self.process(c) for c in expr.children)
      return self.insert(arith.OrIOp(a, b))

    @process.register(gem.MinValue)
    def _expression_min(self, expr):
      a, b = (self.process(c) for c in expr.children)
      return self._binary(a, b, arith.MinimumfOp, arith.MinSIOp)

    @process.register(gem.MaxValue)
    def _expression_max(self, expr):
      a, b = (self.process(c) for c in expr.children)
      return self._binary(a, b, arith.MaximumfOp, arith.MaxSIOp)

    _CMPF = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge", "==": "oeq", "!=": "one"}
    _CMPI = {"<": "slt", "<=": "sle", ">": "sgt", ">=": "sge", "==": "eq", "!=": "ne"}


    @process.register(gem.Comparison)
    def _(self, expr):
      a, b = self._align_scalars(*(self.process(c) for c in expr.children))
      if is_float(a.type):
        return self.insert(arith.CmpfOp(a, b, self._CMPF[expr.operator]))
      return self.insert(arith.CmpiOp(a, b, self._CMPI[expr.operator]))

    @process.register(gem.Conditional)
    def _(self, expr):
      # NOTE: Worth noting that arith.select is eager
      # scf.if may be better if branch leads to div-by-zero
      cond, then, else_ = (self.process(c) for c in expr.children)
      then, else_ = self._align_scalars(then, else_)
      return self.insert(arith.SelectOp(cond, then, else_))

    # TODO: Need to deal with more possible functions 
    @process.register(gem.MathFunction)
    def _(self, expr):
      name = expr.name
      args = [self.process(c) for c in expr.children]
      if name == "abs":
        return self.insert((xmath.AbsFOp if is_float(args[0].type) else xmath.AbsIOp)(args[0]))
    
    # }}} 

    # {{{ misc types

    @process.register
    def _(self, obj: types.NoneType | numbers.Number, /, **kwargs):
        return obj

    # }}}

    @contextlib.contextmanager
    def enter_for(self, index): 
      # Define SSA for extent, if it does not exist 
      if hasattr(index, "extent"):
        extent = index.extent
      elif isinstance(index, numbers.Integral):
        extent = index

      lb = self._const(0, INDEX_TYPE)
      ub = self.get_index_ssa(extent)
      step = self._const(1, INDEX_TYPE)

      for_op = scf.ForOp(lb, ub, step, [], Region(Block(arg_types=[INDEX_TYPE])))
      self.insert(for_op)

      body = for_op.body.block 
      self.symbol_table.push()
      self.symbol_table.define(index, body.args[0])
      self._builder_stack.append(Builder(InsertPoint.at_end(body)))

      yield 

      self.insert(scf.YieldOp())
      self._builder_stack.pop()
      self.symbol_table.pop()
