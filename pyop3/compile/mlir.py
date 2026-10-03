import functools
import numbers
import types
from typing import Any

from immutabledict import immutabledict as idict

import pyop3.axis_tree
import pyop3.expr
import pyop3.insn
import pyop3.node
from pyop3 import utils

from gem import gem, impero as imp 

from xdsl.dialects import arith, func, memref, scf
from xdsl.builder import Builder, InsertPoint
from xdsl.ir import SSAValue, Block, Region, Operation

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    DYNAMIC_INDEX,
    IntegerType,
    IndexType,
    IntegerAttr,
    Float64Type,
    FloatAttr,
    i64, i32, f64,
    MemRefType,
    ModuleOp,
    UnitAttr,
)

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


class MLIRBuilder(pyop3.node.NodeVisitor):
    """Visitor that populates an MLIR Object.

    This class is intended to be subclassed by 'reconstruction' visitors that
    build similar objects.

    """
    def __init__(self, shallow: bool = False, allowed_types: set | None = None) -> None:
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

        # NameVar temporaries already resolved to SSA values live in
        # symbol_table; this records which names are temporaries.
        self._temporaries: set = set()

        self._name_generator = utils.UniqueNameGenerator()
        
        self.shallow = shallow
        super().__init__()

    def visit_path(self, path, **kwargs):
        return path

    def _visit_pathed_mapping(self, mapping, **kwargs):
        return idict({
            self.visit_path(path, **kwargs): self(value, **kwargs)
            for path, value in mapping.items()
        })

    @property
    def builder(self) -> Builder:
        return self._builder_stack[-1]
    
   # {{{ helpers 

    def unique_name(self, prefix: str) -> str:
        return self._name_generator(prefix)

    def insert(self, op: Operation) -> SSAValue:
        """ Inserts an MLIR operation into the block, returning SSA result """
        self.builder.insert(op)
        results = op.results 
        return results[0] if results else None 

   # FIXME: Adjust for GEM stack
    def insert_arg(
            self, 
            arg: Argument, 
            buffer_view: pyop3.expr.IndexedBuffer 
    ) -> None:
        """ Insert args into function definition and symbol table """ 
        assert isinstance(arg, Argument)

        name = arg.name 
        buffer_type = self._buffer_type(arg)

        block = self._entry_block
        memref_ssa = block.insert_arg(buffer_type, len(block.args)) 
        if name: 
            memref_ssa.name_hint = name 

        self.symbol_table.define(buffer_view, memref_ssa)
        return memref_ssa 

    def _const_index(self, value: int) -> SSAValue:
        const_ssa = self.insert(arith.ConstantOp(
            IntegerAttr(int(value), iType)
           )
        )
        return const_ssa


      def collect_gem_name(self, node):
          var = self.unique_name(node.name)
          # FIXME: Need alternative to Subscript
          # if node in self.indices:
          #     indices = self.fetch_multiindex(self.indices[node])
              # if indices:
              #     return p.Subscript(pym, indices)
          return var 

    # }}} 

    @functools.singledispatchmethod
    def process(self, obj: Any, /, **kwargs):
        utils.raise_missing_dispatch_handler(obj)

    # {{{ impero types

    # NOTE: Possible that this is not necessary if Blocks are only nested in ForOps
    @process.register(imp.Block)
    def process_block(self, tree): 
      from itertools import chain
      with enter_block(): 
        chain(*(process(child, ctx) for child in tree.children))

    @process.register(imp.For)
    def process_for(self, tree): 

      extent = tree.index.extent
      assert extent

      # NOTE: Index hopefully tagged with parallel at this stage  
      # if tree.index.is_parallel... 

      with enter_for(extent):
        process(tree.children[0], ctx)

    @process.register(imp.Initialise)
    def process_initialise(self, leaf): 
      ...

    @process.register(imp.Accumulate)
    def process_accumulate(self, leaf): 
      """ Process IndexSum  """ 
    ...

    # NOTE: Are these necessary? How would they present in MLIR?
    # Only of use if not buffer-type, memref changes buffer in-place. If returning scalar value. MLIR would need to have return type defined
    @process.register(imp.Return)
    def process_return(self, leaf):
      ...

    # Same as above. Albeit accumulate must happen. Could just redirect to accumulate. 
    @process.register(imp.ReturnAccumulate)
    def process_returnaccumulate(self, leaf):
      ...

    @process.register(imp.Evaluate) 
    def process_evaluate(self, leaf):
      """ Assign temporary, mapping expr to SSA for re-use """ 
      expr = leaf.expression
      name = self.collect_gem_name(expr)

      if name in self.symbol_table:
        return self.symbol_table[name]

      ssa = self.process(expr) 
      self.symbol_table[name] = ssa 
      return ssa 

    # }}}

    # {{{ gem types 
    
    @process.register(gem.Constant)
    def _(self, leaf):
      ...

    @process.register(gem.Literal)
    def _(self, leaf):
      """ Process gem.Literal which is a tensor-valued constant """
      # Few options here...
      ...

    @process.register(gem.ListTensor)
    def _(self, leaf):
      ...
      
    @process.register(gem.ComponentTensor)
    def _(self, leaf):
      ...

    @process.register(gem.Variable)
    def _(self, leaf):
      ...

    @process.register(gem.Indexed)
    def _(self, leaf):
      ...

    @process.register(gem.VariableIndex)
    def _(self, leaf):
      ...

    @process.register(gem.IndexSum)
    def _(self, leaf):
      ...

    # }}}

    # {{{ gem operations 


    @process.register(gem.Sum)
    def _expression_sum(self, expr):
      ...

    @process.register(gem.Product)
    def _expression_product(self, expr):
      ...

    @process.register(gem.Division)
    def _expression_division(self, expr):
      ...

    @process.register(gem.LogicalAnd)
    def _expression_logicaland(self, expr):
      ...

    # NOTE: Many more... 

    # }}} 

    # {{{ misc types

    @process.register
    def _(self, obj: types.NoneType | numbers.Number, /, **kwargs):
        return obj

    # }}}

    @functools.contextmanager
    def enter_for(self, extent): 
      # Define SSA for extent, if it does not exist 
      if extent not in self.symbol_table:
        extent_ssa = self.process(extent)  
      else: 
        extent_ssa = self.symbol_table[extent]

      lb = self._const_index(0)
      ub = extent_ssa
      step = self._const_index(1) 

      for_op = scf.ForOp(lb, ub, step, [], Region(Block(arg_types=[iType])))

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
    @functools.contextmanager
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


