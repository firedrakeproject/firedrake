from __future__ import annotations

import functools
import contextlib
import numbers
import types
from typing import Any


from immutabledict import immutabledict as idict

from pyop3 import utils

from gem import gem, impero as imp 

from xdsl.dialects import arith, func, memref, scf
from xdsl.builder import Builder, InsertPoint
from xdsl.ir import SSAValue, Block, Region, Operation

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerType,
    IndexType,
    IntegerAttr,
    Float64Type,
    FloatAttr,
    i32, i64, f32, f64,
    MemRefType,
    ModuleOp,
    UnitAttr,
)

INDEX_TYPE = IndexType()
FLOAT_TYPES = (f32, f64)

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
      self.globals: list[Operation] = [] 
      self._global_cache: dict[Any, tuple[str, MemRefType]] = dict()

      self._name_generator = utils.UniqueNameGenerator()
      

    @property
    def builder(self) -> Builder:
      return self._builder_stack[-1]

    def build(self, root, name: str = "imp_kernel", args=()) -> ModuleOp:
      """ 
      Construct the MLIR module, containing a FuncOp, given an Impero tree.

      :arg root: ImperoC tree
      :arg name: Name of generated MLIR kernel
      :arg args: gem.Variable in desired argument order. gem.Variables not provided are appended in order of visitation
      """
      for variable in args:
          self._create_buffer(variable)
      self.process(root)
      self._entry_block.add_op(func.ReturnOp())
      arg_types = tuple(a.type for a in self._entry_block.args)
      fn = func.FuncOp(name, (arg_types, ()), Region(self._entry_block))
      return ModuleOp([*self.globals, fn])
    
   # {{{ helpers 

    def unique_name(self, prefix: str) -> str:
      return self._name_generator(prefix)

    def insert(self, op: Operation) -> SSAValue:
      """ Inserts an MLIR operation into the block, returning SSA result """
      self.builder.insert(op)
      results = op.results 
      return results[0] if results else None 
  
    def create_const(self, value, mlir_type) -> SSAValue:
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
        return self._cast(v, to)

      raise NotImplementedError(f"Casting frm {frm} to {to} is not implemented")

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

      return self._cast(a, target), self._cast(b, target)

    def _binary(self, a, b, float_op, int_op) -> SSAValue:
      a, b = self._align_scalars(a, b)
      return self.insert((float_op if is_float(a.type) else int_op)(a, b))

    def _extent(self, index) -> int:
      """ Return extent for index 
      
      Assuming that this could be an SSA value
      But this could be an SSA value so revisit this. 

      Leaving wrapped as function for above concern.

      """
      extent = index.extent

      if not isinstance(extent, numbers.Integral):
        raise NotImplementedError(f"Index {index} has variable index extent {extent!r}")

      return int(extent)

    # }}} 

    # {{{ buffers and temporaries
    
    @functools.singledispatchmethod
    def _create_buffer(self, node) -> SSAValue:
      utils.raise_missing_dispatch_handler(node)

    @_create_buffer.register(gem.Variable)
    def _(self, node: gem.Variable) -> SSAValue:
      if node.name in self._buffer_args: 
        return self._buffer_args[node.name] 
    
      # NOTE: Assuming that the variables are function arguments.
      # This is a mistake, Variable are just runtime-known values. 
      # FIXME: Need to reconsider how to approach gem.Variable parsing
      assert isinstance(node.shape[0], numbers.Integral)
  
      memref_type = MemRefType(get_mlir_type(node.dtype), node.shape)

      # Insert buffer into end of function argument signature 
      arg = self._entry_block.insert_arg(memref_type, len(self._entry_block.args))
      arg.name_hint = node.name
      self._buffer_args[node.name] = arg 
      return self._buffer_args[node.name]

    @_create_buffer.register(gem.Literal)
    def _(self, node: gem.Literal) -> SSAValue:
      """ Define Literal (tensor-valued constant) as buffer  
      
      I would like to define constant literals in advance of this process. 
      They are known in advance anyway. 
    
      """ 
      if node in self._temporaries:
        return self._temporaries[name] 
      
      arr = node.array 
      mlir_type = get_mlir_type(arr.dtype)
      memref_type = MemRefType(mlir_type, arr.shape)

      name = self.unique_name("literal")
      alloca_op = memref.AllocaOp.get(memref_type, shape=arr.shape)
        
      # NOTE: Insert into top body
      # TODO: Define constants in advance, avoid this ugly portion 
      Builder(InsertPoint.at_start(self._entry_block)).insert(alloca_op)
      self._temporaries[node] = alloca_op.results[0]

      return self._temporaries[node]


    @functools.singledispatchmethod
    def get_index_ssa(self, idx) -> SSAValue:
      utils.raise_missing_dispatch_handler(idx)

    @get_index_ssa.register(numbers.Integral)
    def _(self, idx: numbers.Integral):
      return self._const(idx, INDEX_TYPE)

    @get_index_ssa.register(IndexType)
    def _(self, idx: INDEX_TYPE):
      return self.symbol_table[idx]
      
    @get_index_ssa.register(gem.VariableIndex)
    def _(self, idx: gem.VariableIndex):
      return self._as_index(self.emit(idx.expression))

    @get_index_ssa.register(gem.Node)
    def _(self, idx: gem.Node):
      return self._as_index(self.emit(idx))

    @functools.singledispatchmethod
    def get_address(self, node: Any):
      """(memref, indices) for an lvalue: Variable, or Indexed/Gather of one."""
      utils.raise_missing_dispatch_handler(node)

    @get_address.register(gem.Variable)
    def _(self, node):
      return self._create_buffer(node), []

    @get_address.register(gem.Indexed)
    @get_address.register(gem.Gather)
    def _(self, node): 
      assert isinstance(node.children[0], gem.Variable)
      return (self._create_buffer(node.children[0]), [self.get_index_ssa(i) for i in node.multiindex])  
  

    def _store(self, node, value: SSAValue, accumulate: bool = False) -> None:
      buf, idx = self._address(node)
      elem = buf.type.element_type
      value = self._cast(value, elem)
      if accumulate:
        old = self.insert(memref.LoadOp.get(buf, idx))
        value = self._binary(old, value, arith.AddfOp, arith.AddiOp)
      self.insert(memref.StoreOp.get(value, buf, idx))

    # NOTE: Might avoid a dispatch function at this point 
    def emit(self, expr) -> SSAValue:
      """Value of a GEM expression, reusing an Evaluate/IndexSum temporary if one exists."""
      if isinstance(expr, numbers.Number):
        return self._const(expr, i64 if isinstance(expr, numbers.Integral) else f64)
      if isinstance(expr, gem.Node) and expr in self._temporaries:
        buf, _ = self._temporaries[expr]
        return self.insert(memref.LoadOp.get(buf, self._temp_indices(expr)))
      return self.process(expr)


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

    @index_into.register(gem.Variable)
    @index_into.register(gem.Literal)
    def _(self, aggregate, multiindex) -> SSAValue:
      buf = self._create_buffer(aggregate)
      return self.insert(
        memref.LoadOp.get(
          buf, [self.get_index_ssa(i) for i in multiindex]
        )
      )

    @index_into.register(gem.ListTensor)
    def _(self, aggregate: gem.ListTensor, multiindex) -> SSAValue:
      """ Indexing into ListTensor (stack or list of tensors essentially) """
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

      # TODO: Explain this better for others
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

      with enter_for(tree.index):
        process(tree.children[0], ctx)

    @process.register(imp.Initialise)
    def process_initialise(self, leaf): 
      isum = leaf.indexsum 
      dtype = get_mlir_type(isum.dtype)
      buf = self._alloc_temporary(isum, isum.free_indices, dtype)
      sop = memref.StoreOp.get(self._const(0, dtype), buf, self._temp_indices(isum))
      self.insert(sop)

    @process.register(imp.Accumulate)
    def process_accumulate(self, leaf): 
      """ Process gem.IndexSum  """ 
      isum = leaf.indexsum
      if isum not in self._temporaries:
          raise KeyError(f"Accumulate before Initialise for {isum}")
      buf, _ = self._temporaries[isum]
      idx = self._temp_indices(isum)
      term = self._cast(self.emit(isum.children[0]), buf.type.element_type)
      old = self.insert(memref.LoadOp.get(buf, idx))
      new = self._binary(old, term, arith.AddfOp, arith.AddiOp)
      self.insert(memref.StoreOp.get(new, buf, idx))

    # NOTE: Are these necessary? How would they present in MLIR?
    # Only of use if not buffer-type, memref changes buffer in-place. If returning scalar value. MLIR would need to have return type defined
    @process.register(imp.Return)
    def process_return(self, leaf):
      self._store(leaf.variable, self.emit(leaf.expression))

    # Same as above. Albeit accumulate must happen. Could just redirect to accumulate. 
    # @process.register(imp.ReturnAccumulate)
    # def process_returnaccumulate(self, leaf):
    #   ...

    @process.register(imp.Evaluate) 
    def process_evaluate(self, leaf):
      """ Assign temporary, mapping expr to SSA for re-use """ 
      # Some serious issues in this portion of code to debug
      expr = leaf.expression
      if expr in self._temporaries:
          return
      if expr.shape:
          raise NotImplementedError("Evaluate of a tensor-valued expression")
      value = self.process(expr)  # bypass the temp lookup: this *is* the definition
      buf = self._alloc_temp(expr, expr.free_indices, value.type)
      self.insert(memref.StoreOp.get(value, buf, self._temp_indices(expr)))

    # }}}

    # {{{ gem types 
    
    @process.register(gem.Zero)
    def _(self, leaf):
      assert not leaf.shape
      return self._const(0, get_mlir_type(leaf.dtype))
    
    @process.register(gem.Literal)
    def _(self, leaf):
      """ Process gem.Literal which is a tensor-valued constant """
      if leaf.shape:
        # This is wrong for when we Evaluate a tensor literal
        raise NotImplementedError("Tensor-valued literals must be indexed")
      return self._const(leaf.value, get_mlir_type(leaf.dtype))

    @process.register(gem.Variable)
    def _(self, leaf):
      if leaf.shape:
        raise NotImplementedError(f"Variable {leaf.name} is tensor-valued; it must be indexed")
      return self.insert(memref.LoadOp.get(self._create_buffer(leaf), []))

    @process.register(gem.Index)
    def _(self, leaf):
      return self.symbol_table[leaf] 

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
      a, b = (self.emit(c) for c in expr.children)
      return self._binary(a, b, arith.AddfOp, arith.AddiOp)

    @process.register(gem.Product)
    def _expression_product(self, expr):
      a, b = (self.emit(c) for c in expr.children)
      return self._binary(a, b, arith.MulfOp, arith.MuliOp)

    @process.register(gem.Division)
    def _expression_division(self, expr):
      a, b = (self.emit(c) for c in expr.children)
      return self._binary(a, b, arith.DivfOp, arith.DivSIOp)

    @process.register(gem.FloorDiv)
    def _expression_floordiv(self, expr):
      a, b = (self.emit(c) for c in expr.children)
      a, b = self._align_scalars(a, b)

      assert not is_float(a.type)

      return self.insert(arith.FloorDivSIOp(a, b))

    @process.register(gem.Remainder)
    def _expression_remainder(self, expr):
      a, b = (self.emit(c) for c in expr.children)
      a, b = self._align_scalars(a, b)
      
      assert not is_float(a.type)
       
      return self.insert(arith.RemSIOp(a, b))

    @process.register(gem.LogicalAnd)
    def _expression_logicaland(self, expr):
      a, b = (self.emit(c) for c in expr.children)
      return self.insert(arith.AndIOp(a, b))

    @process.register(gem.LogicalOr)
    def _expression_logicalor(self, expr):
      a, b = (self.emit(c) for c in expr.children)
      return self.insert(arith.OrIOp(a, b))

    @process.register(gem.MinValue)
    def _expression_min(self, expr):
      a, b = (self.emit(c) for c in expr.children)
      return self._binary(a, b, arith.MinimumfOp, arith.MinSIOp)

    @process.register(gem.MaxValue)
    def _expression_max(self, expr):
      a, b = (self.emit(c) for c in expr.children)
      return self._binary(a, b, arith.MaximumfOp, arith.MaxSIOp)

    _CMPF = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge", "==": "oeq", "!=": "one"}
    _CMPI = {"<": "slt", "<=": "sle", ">": "sgt", ">=": "sge", "==": "eq", "!=": "ne"}


    @process.register(gem.Comparison)
    def _(self, expr):
      a, b = self._align_scalars(*(self.emit(c) for c in expr.children))
      if is_float(a.type):
        return self.insert(arith.CmpfOp(a, b, self._CMPF[expr.operator]))
      return self.insert(arith.CmpiOp(a, b, self._CMPI[expr.operator]))

    @process.register(gem.Conditional)
    def _(self, expr):
      # NOTE: Worth noting that arith.select is eager
      # scf.if may be better if branch leads to div-by-zero
      cond, then, else_ = (self.emit(c) for c in expr.children)
      then, else_ = self._align_scalars(then, else_)
      return self.insert(arith.SelectOp(cond, then, else_))

    # }}} 

    # {{{ misc types

    @process.register
    def _(self, obj: types.NoneType | numbers.Number, /, **kwargs):
        return obj

    # }}}

    @contextlib.contextmanager
    def enter_for(self, extent): 
      # Define SSA for extent, if it does not exist 

      extent = self._extent(index)
      lb = self._const(0, INDEX_TYPE)
      ub = self._const(extent, INDEX_TYPE)
      step = self._const(1, INDEX_TYPE)

      for_op = scf.ForOp(lb, ub, step, [], Region(Block(arg_types=[INDEX_TYPE])))
      self.insert(for_op)

      body = for_op.body.block 
      self.symbol_table.push()
      self.symbol_table.define(extent, body.args[0])
      self._builder_stack.append(Builder(InsertPoint.at_end(body)))

      yield 

      self.insert(scf.YieldOp())
      self._builder_stack.pop()
      self.symbol_table.pop()


    # TODO: Factorise with iterative loop
    @contextlib.contextmanager
    def enter_block(self): 
      block = Block()
      builder = Builder(InsertPoint.at_end(block))
      
      self._builder_stack.append(builder) 
      self.symbol_table.push()

      yield 

      # NOTE: ReturnOp??

      # Attach Block to parent Block (parent in builder stack) 
      populated_block = self._builder_stack.pop()
      self.symbol_table.pop()
      self.insert(populated_block) 


