module {
  func.func @pyop3_loop(%arg0: tensor<?xf64>, %arg1: tensor<?xi32>, %arg2: tensor<?xi32>, %arg3: tensor<?xf64>, %arg4: tensor<?xi32>, %arg5: tensor<?xi32>) -> tensor<?xf64> attributes {llvm.emit_c_interface} {
    %c16 = arith.constant 16 : index
    %cst = arith.constant 2.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg6 = %c0 to %c16 step %c1 iter_args(%arg7 = %arg0) -> (tensor<?xf64>) {
      %extracted = tensor.extract %arg4[%arg6] : tensor<?xi32>
      %1 = arith.index_cast %extracted : i32 to index
      %extracted_0 = tensor.extract %arg1[%1] : tensor<?xi32>
      %2 = arith.index_cast %extracted_0 : i32 to index
      %extracted_1 = tensor.extract %arg3[%2] : tensor<?xf64>
      %3 = arith.mulf %extracted_1, %cst : f64
      %inserted = tensor.insert %3 into %arg7[%2] : tensor<?xf64>
      scf.yield %inserted : tensor<?xf64>
    }
    return %0 : tensor<?xf64>
  }
}

