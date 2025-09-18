func.func @mykernel(%myvar3:tensor<6xf64>, %myvar4:tensor<6xf64>, %out:tensor<6xf64>) -> tensor<6xf64>
{
%myvar0 = arith.constant dense<[[-0.04820838, 0.79548023, 0.19283351,-0.04820838, 0.19283351,-0.08473049],
 [-0.08473049, 0.19283351, 0.19283351,-0.04820838, 0.79548023,-0.04820838],
 [-0.04820838, 0.19283351, 0.79548023,-0.08473049, 0.19283351,-0.04820838],
 [ 0.51763234, 0.29921523, 0.29921523,-0.07480381, 0.03354481,-0.07480381],
 [-0.07480381, 0.29921523, 0.03354481, 0.51763234, 0.29921523,-0.07480381],
 [-0.07480381, 0.03354481, 0.29921523,-0.07480381, 0.29921523, 0.51763234]]> : tensor<6x6xf64>
%myvar1 = arith.constant dense<[0.11169079,0.11169079,0.11169079,0.05497587,0.05497587,0.05497587]> : tensor<6xf64>
%myvar2 = arith.constant -1.0 : f64
%dummy = tensor.empty() : tensor<6x6x6xf64>
%myresult = linalg.generic
    {
        indexing_maps = [affine_map<(A, B, C) -> (A, B, C)>, affine_map<(A, B, C) -> (A)>],
        iterator_types = ["parallel", "reduction", "reduction"]
    }
    ins(%dummy: tensor<6x6x6xf64>)
    outs(%out: tensor<6xf64>)
    {
        ^bb0(%bdummy: f64, %bout : f64) :
        %myvar6 = linalg.index 1 : index
    %myvar7 = linalg.index 0 : index
    %myvar5 = tensor.extract %myvar0[%myvar6, %myvar7] : tensor<6x6xf64>
    %myvar9 = linalg.index 1 : index
    %myvar8 = tensor.extract %myvar1[%myvar9] : tensor<6xf64>
    %myvar11 = arith.constant 0 : index
    %myvar10 = tensor.extract %myvar3[%myvar11] : tensor<6xf64>
    %myvar12 = arith.mulf %myvar2, %myvar10 : f64
    %myvar14 = arith.constant 2 : index
    %myvar13 = tensor.extract %myvar3[%myvar14] : tensor<6xf64>
    %myvar15 = arith.addf %myvar12, %myvar13 : f64
    %myvar17 = arith.constant 1 : index
    %myvar16 = tensor.extract %myvar3[%myvar17] : tensor<6xf64>
    %myvar18 = arith.mulf %myvar2, %myvar16 : f64
    %myvar20 = arith.constant 5 : index
    %myvar19 = tensor.extract %myvar3[%myvar20] : tensor<6xf64>
    %myvar21 = arith.addf %myvar18, %myvar19 : f64
    %myvar22 = arith.mulf %myvar15, %myvar21 : f64
    %myvar24 = arith.constant 0 : index
    %myvar23 = tensor.extract %myvar3[%myvar24] : tensor<6xf64>
    %myvar25 = arith.mulf %myvar2, %myvar23 : f64
    %myvar27 = arith.constant 4 : index
    %myvar26 = tensor.extract %myvar3[%myvar27] : tensor<6xf64>
    %myvar28 = arith.addf %myvar25, %myvar26 : f64
    %myvar30 = arith.constant 1 : index
    %myvar29 = tensor.extract %myvar3[%myvar30] : tensor<6xf64>
    %myvar31 = arith.mulf %myvar2, %myvar29 : f64
    %myvar33 = arith.constant 3 : index
    %myvar32 = tensor.extract %myvar3[%myvar33] : tensor<6xf64>
    %myvar34 = arith.addf %myvar31, %myvar32 : f64
    %myvar35 = arith.mulf %myvar28, %myvar34 : f64
    %myvar36 = arith.mulf %myvar2, %myvar35 : f64
    %myvar37 = arith.addf %myvar22, %myvar36 : f64
    %myvar38 = math.absf %myvar37 : f64
    %myvar39 = arith.mulf %myvar8, %myvar38 : f64
    %myvar41 = linalg.index 2 : index
    %myvar42 = linalg.index 1 : index
    %myvar40 = tensor.extract %myvar0[%myvar41, %myvar42] : tensor<6x6xf64>
    %myvar46 = arith.constant 1 : index
    %myvar44 = linalg.index 2 : index
    %myvar45 = arith.muli %myvar44, %myvar46 : index
    %myvar47 = arith.constant 0 : index
    %myvar48 = arith.addi %myvar47, %myvar45 : index
    %myvar43 = tensor.extract %myvar4[%myvar48] : tensor<6xf64>
    %myvar49 = arith.mulf %myvar40, %myvar43 : f64
    %myvar50 = arith.mulf %myvar39, %myvar49 : f64
    %myvar51 = arith.mulf %myvar5, %myvar50 : f64
        %inc = arith.addf %bout, %myvar51: f64
        linalg.yield %inc : f64
    } -> tensor<6xf64>
func.return %myresult : tensor<6xf64>
}