from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func, scf, tensor
from xdsl.dialects.builtin import (
    FloatAttr,
    IndexType,
    ModuleOp,
    TensorType,
    f32,
    i32,
)
from xdsl.ir import Block, Region

index = IndexType()

N = 128
t_f = TensorType(f32, [N])   # dat_2, dat_3   (float data)
t_i = TensorType(i32, [N])   # idat_0, idat_2 (index data)

fn_block = Block(arg_types=[t_f, t_f, t_i, t_i])

with ImplicitBuilder(fn_block) as (dat_2, dat_3, idat_0, idat_2):
    c0   = arith.ConstantOp.from_int_and_width(0, index)
    c1   = arith.ConstantOp.from_int_and_width(1, index)
    n    = arith.ConstantOp.from_int_and_width(N, index)
    two  = arith.ConstantOp(FloatAttr(2.0, f32))

    body = Block(arg_types=[index, t_f])
    with ImplicitBuilder(body) as (i2, acc):
        # k = idat_2[i_2]
        k     = tensor.ExtractOp(idat_2, [i2], i32)
        k_idx = arith.IndexCastOp(k.result, index)

        # j = idat_0[k]
        j     = tensor.ExtractOp(idat_0, [k_idx.result], i32)
        j_idx = arith.IndexCastOp(j.result, index)

        # v = dat_3[j]
        v     = tensor.ExtractOp(dat_3, [j_idx.result], f32)

        # r = 2.0 * v
        r     = arith.MulfOp(two.result, v.result)

        # dat_2[j] = r  (value semantics -> produces new tensor)
        new   = tensor.InsertOp(r.result, acc, [j_idx.result])

        scf.YieldOp(new.result)

    loop = scf.ForOp(
        lb=c0.result,
        ub=n.result,
        step=c1.result,
        iter_args=[dat_2],
        body=Region(body),
    )

    func.ReturnOp(loop.results[0])

fn = func.FuncOp(
    "indirect_scale",
    ((t_f, t_f, t_i, t_i), (t_f,)),
    Region(fn_block),
)

module = ModuleOp([fn])
print(module)
