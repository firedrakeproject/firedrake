builtin.module {
  func.func @pyop3_loop(%0: tensor<?xf64>, %1: tensor<?xi32>, %2: tensor<?xi32>, %3: tensor<?xf64>, %4: tensor<?xi32>, %5: tensor<?xi32>) {
    %6 = arith.constant 0 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.constant 18 : i32
    %9 = arith.index_cast %8 : i32 to index
    %10 = arith.constant 1 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = scf.for %13 = %7 to %9 step %11 iter_args(%14 = %0) -> (tensor<?xf64>) {
      %15 = arith.constant 0 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.constant 0 : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.constant 1 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = scf.for %22 = %16 to %18 step %20 iter_args(%23 = %14) -> (tensor<?xf64>) {
        %24 = arith.constant 2 : i32
        %25 = arith.constant 0 : i32
        %26 = arith.constant 1 : i32
        %27 = arith.constant 0 : i32
        %28 = arith.constant 1 : i32
        %29 = arith.constant 0 : i32
        %30 = arith.constant 1 : i32
        %31 = arith.muli %30, %13 : i32
        %32 = arith.addi %29, %31 : i32
        %33 = arith.index_cast %32 : i32 to index
        %34 = tensor.extract %2[%33] : tensor<?xi32>
        %35 = arith.muli %28, %34 : i32
        %36 = arith.addi %27, %35 : i32
        %37 = arith.index_cast %36 : i32 to index
        %38 = tensor.extract %1[%37] : tensor<?xi32>
        %39 = arith.addi %38, %22 : i32
        %40 = arith.muli %26, %39 : i32
        %41 = arith.addi %25, %40 : i32
        %42 = arith.index_cast %41 : i32 to index
        %43 = tensor.extract %3[%42] : tensor<?xf64>
        %44 = arith.muli %24, %43 : i32
        %45 = arith.constant 0 : i32
        %46 = arith.constant 1 : i32
        %47 = arith.constant 0 : i32
        %48 = arith.constant 1 : i32
        %49 = arith.constant 0 : i32
        %50 = arith.constant 1 : i32
        %51 = arith.muli %50, %13 : i32
        %52 = arith.addi %49, %51 : i32
        %53 = arith.index_cast %52 : i32 to index
        %54 = tensor.extract %2[%53] : tensor<?xi32>
        %55 = arith.muli %48, %54 : i32
        %56 = arith.addi %47, %55 : i32
        %57 = arith.index_cast %56 : i32 to index
        %58 = tensor.extract %1[%57] : tensor<?xi32>
        %59 = arith.addi %58, %22 : i32
        %60 = arith.muli %46, %59 : i32
        %61 = arith.addi %45, %60 : i32
        %62 = arith.index_cast %61 : i32 to index
        %63 = tensor.insert %44 into %23[%62] : tensor<?xf64>
        scf.yield %63 : tensor<?xf64>
      }
      scf.yield %21 : tensor<?xf64>
    }
    %64 = arith.constant 0 : i32
    %65 = arith.index_cast %64 : i32 to index
    %66 = arith.constant 16 : i32
    %67 = arith.index_cast %66 : i32 to index
    %68 = arith.constant 1 : i32
    %69 = arith.index_cast %68 : i32 to index
    %70 = scf.for %71 = %65 to %67 step %69 iter_args(%72 = %12) -> (tensor<?xf64>) {
      %73 = arith.constant 2 : i32
      %74 = arith.constant 0 : i32
      %75 = arith.constant 1 : i32
      %76 = arith.constant 0 : i32
      %77 = arith.constant 1 : i32
      %78 = arith.constant 0 : i32
      %79 = arith.constant 1 : i32
      %80 = arith.muli %79, %71 : i32
      %81 = arith.addi %78, %80 : i32
      %82 = arith.index_cast %81 : i32 to index
      %83 = tensor.extract %4[%82] : tensor<?xi32>
      %84 = arith.muli %77, %83 : i32
      %85 = arith.addi %76, %84 : i32
      %86 = arith.index_cast %85 : i32 to index
      %87 = tensor.extract %1[%86] : tensor<?xi32>
      %88 = arith.constant 0 : i32
      %89 = arith.addi %87, %88 : i32
      %90 = arith.muli %75, %89 : i32
      %91 = arith.addi %74, %90 : i32
      %92 = arith.index_cast %91 : i32 to index
      %93 = tensor.extract %3[%92] : tensor<?xf64>
      %94 = arith.muli %73, %93 : i32
      %95 = arith.constant 0 : i32
      %96 = arith.constant 1 : i32
      %97 = arith.constant 0 : i32
      %98 = arith.constant 1 : i32
      %99 = arith.constant 0 : i32
      %100 = arith.constant 1 : i32
      %101 = arith.muli %100, %71 : i32
      %102 = arith.addi %99, %101 : i32
      %103 = arith.index_cast %102 : i32 to index
      %104 = tensor.extract %4[%103] : tensor<?xi32>
      %105 = arith.muli %98, %104 : i32
      %106 = arith.addi %97, %105 : i32
      %107 = arith.index_cast %106 : i32 to index
      %108 = tensor.extract %1[%107] : tensor<?xi32>
      %109 = arith.constant 0 : i32
      %110 = arith.addi %108, %109 : i32
      %111 = arith.muli %96, %110 : i32
      %112 = arith.addi %95, %111 : i32
      %113 = arith.index_cast %112 : i32 to index
      %114 = tensor.insert %94 into %72[%113] : tensor<?xf64>
      scf.yield %114 : tensor<?xf64>
    }
    %115 = arith.constant 0 : i32
    %116 = arith.index_cast %115 : i32 to index
    %117 = arith.constant 33 : i32
    %118 = arith.index_cast %117 : i32 to index
    %119 = arith.constant 1 : i32
    %120 = arith.index_cast %119 : i32 to index
    %121 = scf.for %122 = %116 to %118 step %120 iter_args(%123 = %70) -> (tensor<?xf64>) {
      %124 = arith.constant 0 : i32
      %125 = arith.index_cast %124 : i32 to index
      %126 = arith.constant 0 : i32
      %127 = arith.index_cast %126 : i32 to index
      %128 = arith.constant 1 : i32
      %129 = arith.index_cast %128 : i32 to index
      %130 = scf.for %131 = %125 to %127 step %129 iter_args(%132 = %123) -> (tensor<?xf64>) {
        %133 = arith.constant 2 : i32
        %134 = arith.constant 0 : i32
        %135 = arith.constant 1 : i32
        %136 = arith.constant 0 : i32
        %137 = arith.constant 1 : i32
        %138 = arith.constant 0 : i32
        %139 = arith.constant 1 : i32
        %140 = arith.muli %139, %122 : i32
        %141 = arith.addi %138, %140 : i32
        %142 = arith.index_cast %141 : i32 to index
        %143 = tensor.extract %5[%142] : tensor<?xi32>
        %144 = arith.muli %137, %143 : i32
        %145 = arith.addi %136, %144 : i32
        %146 = arith.index_cast %145 : i32 to index
        %147 = tensor.extract %1[%146] : tensor<?xi32>
        %148 = arith.addi %147, %131 : i32
        %149 = arith.muli %135, %148 : i32
        %150 = arith.addi %134, %149 : i32
        %151 = arith.index_cast %150 : i32 to index
        %152 = tensor.extract %3[%151] : tensor<?xf64>
        %153 = arith.muli %133, %152 : i32
        %154 = arith.constant 0 : i32
        %155 = arith.constant 1 : i32
        %156 = arith.constant 0 : i32
        %157 = arith.constant 1 : i32
        %158 = arith.constant 0 : i32
        %159 = arith.constant 1 : i32
        %160 = arith.muli %159, %122 : i32
        %161 = arith.addi %158, %160 : i32
        %162 = arith.index_cast %161 : i32 to index
        %163 = tensor.extract %5[%162] : tensor<?xi32>
        %164 = arith.muli %157, %163 : i32
        %165 = arith.addi %156, %164 : i32
        %166 = arith.index_cast %165 : i32 to index
        %167 = tensor.extract %1[%166] : tensor<?xi32>
        %168 = arith.addi %167, %131 : i32
        %169 = arith.muli %155, %168 : i32
        %170 = arith.addi %154, %169 : i32
        %171 = arith.index_cast %170 : i32 to index
        %172 = tensor.insert %153 into %132[%171] : tensor<?xf64>
        scf.yield %172 : tensor<?xf64>
      }
      scf.yield %130 : tensor<?xf64>
    }
    func.return
  }
}
