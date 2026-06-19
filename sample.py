# Basic functionality
# Make MLIR in xDSL that adds two unranked tensor arrays
# Take this MLIR and lower to LLVM in xDSL
# Compile JIT with llvmlite? 

import sys
from xdsl.dialects import arith, func, memref, scf, tensor, linalg
from xdsl.dialects.arith import ConstantOp, AddfOp
from xdsl.dialects.tensor import DimOp, EmptyOp
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    ModuleOp,
    IndexType,
    IntegerAttr,
    f64,
    TensorType,
    ArrayAttr,
    AffineMap,
    AffineMapAttr,
    AffineDimExpr,
    UnitAttr
)
from xdsl.dialects.linalg import (
    IteratorTypeAttr,
    YieldOp
)

from xdsl.ir import Block, Region
from xdsl.context import Context
from xdsl.printer import Printer

def build_array_add(n: int) -> ModuleOp:
    tensor_type = TensorType(f64, [DYNAMIC_INDEX])
 
    identity_1d = AffineMap(num_dims=1, num_symbols=0, results=(AffineDimExpr(0),))
    identity_attr = AffineMapAttr(identity_1d)
 
    parallel = IteratorTypeAttr.parallel()
 
    func_block = Block(arg_types=[tensor_type, tensor_type, tensor_type])
    a, b, out = func_block.args
 
    c0 = ConstantOp(IntegerAttr(0, IndexType()))
    func_block.add_op(c0)
 
    body_block = Block(arg_types=[f64, f64, f64])
    x, y, _z = body_block.args
 
    add = AddfOp(x, y)         
    body_block.add_op(add)
 
    body_block.add_op(YieldOp(add.result))  
 
    generic = linalg.GenericOp(
        inputs=[a, b],
        outputs=[out],
        body=Region([body_block]),
        indexing_maps=[identity_attr, identity_attr, identity_attr],
        iterator_types=[parallel],
        result_types=[tensor_type],
    )
    func_block.add_op(generic)
 
    func_block.add_op(func.ReturnOp())
 
    func_region = Region([func_block])
    func_op = func.FuncOp(
        "add",
        ([tensor_type, tensor_type, tensor_type], []),
        func_region,
    )

    func_op.attributes["llvm.emit_c_interface"] = UnitAttr()
 
    return ModuleOp([func_op])

def emit_mlir(module: ModuleOp) -> str:
    """Return the MLIR text representation of a module."""
    import io
    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue()

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    # Register dialects so xDSL can verify the IR
    ctx = Context()
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(linalg.Linalg)
    ctx.load_dialect(tensor.Tensor)

    module = build_array_add(n)
    mlir = emit_mlir(module)
    print(mlir)

    
