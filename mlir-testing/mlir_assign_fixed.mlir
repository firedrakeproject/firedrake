// 62
builtin.module {
  func.func @pyop3_loop(%0: tensor<?xf64>, %1: tensor<?xi32>, %2: tensor<?xi32>, %3: tensor<?xf64>, %4: tensor<?xi32>, %5: tensor<?xi32>) -> tensor<?xf64> attributes {llvm.emit_c_interface} {
    %7 = arith.constant 0 : index
    %9 = arith.constant 18 : index
    %11 = arith.constant 1 : index
    %12 = scf.for %13 = %7 to %9 step %11 iter_args(%14 = %0) -> (tensor<?xf64>) {
      %16 = arith.constant 0 : index
      %18 = arith.constant 0 : index 
      %20 = arith.constant 1 : index
      %21 = scf.for %22 = %16 to %18 step %20 iter_args(%23 = %14) -> (tensor<?xf64>) {
        %24 = arith.constant 2. : f64
        %25 = arith.constant 0 : index
        %26 = arith.constant 1 : index
        %27 = arith.constant 0 : index
        %28 = arith.constant 1 : index
        %29 = arith.constant 0 : index
        %30 = arith.constant 1 : index
        %31 = arith.muli %30, %13 : index
        %32 = arith.addi %29, %31 : index
        %34 = tensor.extract %2[%32] : tensor<?xi32>
        %15 = arith.index_cast %34 : i32 to index
        %35 = arith.muli %28, %15 : index
        %36 = arith.addi %27, %35 : index
        %38 = tensor.extract %1[%36] : tensor<?xi32>
        %17 = arith.index_cast %38 : i32 to index
        %39 = arith.addi %17, %22 : index
        %40 = arith.muli %26, %39 : index
        %41 = arith.addi %25, %40 : index
        %43 = tensor.extract %3[%41] : tensor<?xf64>
        %44 = arith.mulf %24, %43 : f64
        %45 = arith.constant 0 : index
        %46 = arith.constant 1 : index
        %47 = arith.constant 0 : index
        %48 = arith.constant 1 : index
        %49 = arith.constant 0 : index
        %50 = arith.constant 1 : index
        %51 = arith.muli %50, %13 : index
        %52 = arith.addi %49, %51 : index
        %54 = tensor.extract %2[%52] : tensor<?xi32>
        %19 = arith.index_cast %54 : i32 to index
        %55 = arith.muli %48, %19 : index
        %56 = arith.addi %47, %55 : index
        %58 = tensor.extract %1[%56] : tensor<?xi32>
        %57 = arith.index_cast %58 : i32 to index
        %59 = arith.addi %57, %22 : index
        %60 = arith.muli %46, %59 : index
        %61 = arith.addi %45, %60 : index
        %63 = tensor.insert %44 into %23[%61] : tensor<?xf64>
        scf.yield %63 : tensor<?xf64>
      }
      scf.yield %21 : tensor<?xf64>
    }
    %65 = arith.constant 0 : index
    %67 = arith.constant 16 : index
    %69 = arith.constant 1 : index
    %70 = scf.for %71 = %65 to %67 step %69 iter_args(%72 = %12) -> (tensor<?xf64>) {
      %73 = arith.constant 2. : f64
      %74 = arith.constant 0 : index
      %75 = arith.constant 1 : index
      %76 = arith.constant 0 : index
      %77 = arith.constant 1 : index
      %78 = arith.constant 0 : index
      %79 = arith.constant 1 : index
      %80 = arith.muli %79, %71 : index
      %81 = arith.addi %78, %80 : index
      %83 = tensor.extract %4[%81] : tensor<?xi32>
      %82 = arith.index_cast %83 : i32 to index
      %84 = arith.muli %77, %82 : index
      %85 = arith.addi %76, %84 : index
      %87 = tensor.extract %1[%85] : tensor<?xi32>
      %86 = arith.index_cast %87 : i32 to index
      %88 = arith.constant 0 : index
      %89 = arith.addi %86, %88 : index
      %90 = arith.muli %75, %89 : index
      %91 = arith.addi %74, %90 : index
      %93 = tensor.extract %3[%91] : tensor<?xf64>
      %94 = arith.mulf %73, %93 : f64
      %95 = arith.constant 0 : index
      %96 = arith.constant 1 : index
      %97 = arith.constant 0 : index
      %98 = arith.constant 1 : index
      %99 = arith.constant 0 : index
      %100 = arith.constant 1 : index
      %101 = arith.muli %100, %71 : index
      %102 = arith.addi %99, %101 : index
      %104 = tensor.extract %4[%102] : tensor<?xi32>
      %92 = arith.index_cast %104 : i32 to index
      %105 = arith.muli %98, %92 : index
      %106 = arith.addi %97, %105 : index
      %108 = tensor.extract %1[%106] : tensor<?xi32>
      %107 = arith.index_cast %108 : i32 to index
      %109 = arith.constant 0 : index
      %110 = arith.addi %107, %109 : index
      %111 = arith.muli %96, %110 : index
      %112 = arith.addi %95, %111 : index
      %114 = tensor.insert %94 into %72[%112] : tensor<?xf64>
      scf.yield %114 : tensor<?xf64>
    }
    %116 = arith.constant 0 : index 
    %118 = arith.constant 33 : index 
    %120 = arith.constant 1 : index 
    %121 = scf.for %122 = %116 to %118 step %120 iter_args(%123 = %70) -> (tensor<?xf64>) {
      %125 = arith.constant 0 : index 
      %127 = arith.constant 0 : index 
      %129 = arith.constant 1 : index 
      %130 = scf.for %131 = %125 to %127 step %129 iter_args(%132 = %123) -> (tensor<?xf64>) {
        %133 = arith.constant 2. : f64
        %134 = arith.constant 0 : index
        %135 = arith.constant 1 : index
        %136 = arith.constant 0 : index
        %137 = arith.constant 1 : index
        %138 = arith.constant 0 : index
        %139 = arith.constant 1 : index
        %140 = arith.muli %139, %122 : index
        %141 = arith.addi %138, %140 : index
        %143 = tensor.extract %5[%141] : tensor<?xi32>
        %142 = arith.index_cast %143 : i32 to index
        %144 = arith.muli %137, %142 : index
        %145 = arith.addi %136, %144 : index
        %147 = tensor.extract %1[%145] : tensor<?xi32>
        %146 = arith.index_cast %147 : i32 to index
        %148 = arith.addi %146, %131 : index
        %149 = arith.muli %135, %148 : index
        %150 = arith.addi %134, %149 : index
        %152 = tensor.extract %3[%150] : tensor<?xf64>
        %153 = arith.mulf %133, %152 : f64
        %154 = arith.constant 0 : index
        %155 = arith.constant 1 : index
        %156 = arith.constant 0 : index
        %157 = arith.constant 1 : index
        %158 = arith.constant 0 : index
        %159 = arith.constant 1 : index
        %160 = arith.muli %159, %122 : index
        %161 = arith.addi %158, %160 : index
        %163 = tensor.extract %5[%161] : tensor<?xi32>
        %162 = arith.index_cast %163 : i32 to index
        %164 = arith.muli %157, %162 : index
        %165 = arith.addi %156, %164 : index
        %167 = tensor.extract %1[%165] : tensor<?xi32>
        %166 = arith.index_cast %167 : i32 to index
        %168 = arith.addi %166, %131 : index
        %169 = arith.muli %155, %168 : index
        %170 = arith.addi %154, %169 : index
        %172 = tensor.insert %153 into %132[%170] : tensor<?xf64>
        scf.yield %172 : tensor<?xf64>
      }
      scf.yield %130 : tensor<?xf64>
    }
    func.return %121 : tensor<?xf64> 
  }
}
