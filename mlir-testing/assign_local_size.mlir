builtin.module {
    func.func @pyop3_loop(
        %dat_0: tensor<?xf64>, 
        %dat_1: tensor<?xf64>,
        %idat_0: tensor<?xi32>,
        %idat_1: tensor<?xi32>,
        %idat_2: tensor<?xi32>,
        %idat_3: tensor<?xi32>
    ) -> tensor<?xf64> {

    %c0 = arith.constant 0 : index // iter var
    %c1 = arith.constant 1 : index // iter var
    %c17 = arith.constant 17 : index // 
    %c15 = arith.constant 15 : index
    %c32 = arith.constant 32 : index
    %c2f = arith.constant 2.0 : f64

    scf.for %i_0 = %c0 to %c17 step %c1 iter_args() -> () {}

    %f2 = scf.for %i_2 = %c0 to %c15 step %c1 iter_args(%dat_it = %dat_0) -> (tensor<?xf64>) {
        %e1 = tensor.extract %idat_2[%i_2] : tensor<?xi32>
        %ii_2 = arith.index_cast %e1 : i32 to index
        %e2 = tensor.extract %idat_0[%ii_2] : tensor<?xi32>
        %iii_2 = arith.index_cast %e2 : i32 to index
        %v1 = tensor.extract %dat_1[%iii_2] : tensor<?xf64>
        %v2 = arith.mulf %c2f, %v1 : f64
        %res = tensor.insert %v2 into %dat_it[%iii_2] : tensor<?xf64>
        scf.yield %res : tensor<?xf64>
    }

    scf.for %i_3 = %c0 to %c32 step %c1 iter_args() -> () {}

    func.return %f2 : tensor<?xf64>
    }
}
