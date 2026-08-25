; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @pyop3_loop(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, ptr %25, ptr %26, i64 %27, i64 %28, i64 %29) {
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %20, 0
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, ptr %21, 1
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, i64 %22, 2
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, i64 %23, 3, 0
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, i64 %24, 4, 0
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %15, 0
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, ptr %16, 1
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, i64 %17, 2
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 %18, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, i64 %19, 4, 0
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %42 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, ptr %6, 1
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %42, i64 %7, 2
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, i64 %8, 3, 0
  %45 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, i64 %9, 4, 0
  %46 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %47 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, ptr %1, 1
  %48 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %47, i64 %2, 2
  %49 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, i64 %3, 3, 0
  %50 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %49, i64 %4, 4, 0
  br label %51

51:                                               ; preds = %54, %30
  %52 = phi i64 [ %85, %54 ], [ 0, %30 ]
  %53 = icmp slt i64 %52, 16
  br i1 %53, label %54, label %86

54:                                               ; preds = %51
  %55 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, 1
  %56 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, 2
  %57 = getelementptr i32, ptr %55, i64 %56
  %58 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, 4, 0
  %59 = mul nuw nsw i64 %52, %58
  %60 = getelementptr inbounds nuw i32, ptr %57, i64 %59
  %61 = load i32, ptr %60, align 4
  %62 = sext i32 %61 to i64
  %63 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, 1
  %64 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, 2
  %65 = getelementptr i32, ptr %63, i64 %64
  %66 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, 4, 0
  %67 = mul nuw nsw i64 %62, %66
  %68 = getelementptr inbounds nuw i32, ptr %65, i64 %67
  %69 = load i32, ptr %68, align 4
  %70 = sext i32 %69 to i64
  %71 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 1
  %72 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 2
  %73 = getelementptr double, ptr %71, i64 %72
  %74 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 4, 0
  %75 = mul nuw nsw i64 %70, %74
  %76 = getelementptr inbounds nuw double, ptr %73, i64 %75
  %77 = load double, ptr %76, align 8
  %78 = fmul double %77, 2.000000e+00
  %79 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 1
  %80 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 2
  %81 = getelementptr double, ptr %79, i64 %80
  %82 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 4, 0
  %83 = mul nuw nsw i64 %70, %82
  %84 = getelementptr inbounds nuw double, ptr %81, i64 %83
  store double %78, ptr %84, align 8
  %85 = add i64 %52, 1
  br label %51

86:                                               ; preds = %51
  %87 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, 3
  %88 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %87, ptr %88, align 4
  %89 = getelementptr [1 x i64], ptr %88, i32 0, i64 0
  %90 = load i64, ptr %89, align 4
  %91 = getelementptr double, ptr null, i64 %90
  %92 = ptrtoint ptr %91 to i64
  %93 = call ptr @malloc(i64 %92)
  %94 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %93, 0
  %95 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %94, ptr %93, 1
  %96 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %95, i64 0, 2
  %97 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, i64 %90, 3, 0
  %98 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %97, i64 1, 4, 0
  %99 = call ptr @llvm.stacksave.p0()
  %100 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, ptr %100, align 8
  %101 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %100, 1
  %102 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %98, ptr %102, align 8
  %103 = insertvalue { i64, ptr } { i64 1, ptr poison }, ptr %102, 1
  %104 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %101, ptr %104, align 8
  %105 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %103, ptr %105, align 8
  call void @memrefCopy(i64 8, ptr %104, ptr %105)
  call void @llvm.stackrestore.p0(ptr %99)
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %98
}

define void @_mlir_ciface_pyop3_loop(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6) {
  %8 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, 0
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, 1
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, 2
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, 3, 0
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, 4, 0
  %14 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 0
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 1
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 2
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 3, 0
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 4, 0
  %20 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 0
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 1
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 2
  %24 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 3, 0
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, 4, 0
  %26 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %4, align 8
  %27 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, 0
  %28 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, 1
  %29 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, 2
  %30 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, 3, 0
  %31 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, 4, 0
  %32 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %5, align 8
  %33 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 0
  %34 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 1
  %35 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 2
  %36 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 3, 0
  %37 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, 4, 0
  %38 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %6, align 8
  %39 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 0
  %40 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 1
  %41 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 2
  %42 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 3, 0
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, 4, 0
  %44 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @pyop3_loop(ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr %21, ptr %22, i64 %23, i64 %24, i64 %25, ptr %27, ptr %28, i64 %29, i64 %30, i64 %31, ptr %33, ptr %34, i64 %35, i64 %36, i64 %37, ptr %39, ptr %40, i64 %41, i64 %42, i64 %43)
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
