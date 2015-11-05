#!/bin/bash

# Mesh loop
for m in 1100.0
do
    # SEQUENTIAL + MPI
    export SLOPE_BACKEND=SEQUENTIAL
    for np in 1, 2, 4
    do
        # Non-tiled tests
        mpirun --bind-to-core -np $np python wave_explicit_fusion.py --mesh-size $m --num-unroll 0
        # Tiled tests
        for ts in 100 200 300 400 500 600 700 800 1000 1500
        do
            mpirun --bind-to-core -np $np python wave_explicit_fusion.py --mesh-size $m --tile-size $ts
        done
        
    done

    #######

    # OMP
    export KMP_AFFINITY=scatter
    export SLOPE_BACKEND=OMP
    for nt in 1, 2, 4
    do
        export OMP_NUM_THREADS=$nt
        # Non-tiled tests
        python wave_explicit_fusion.py --mesh-size $m --num-unroll 0
        # Tiled tests
        for ts in 100 200 300 400 500 600 700 800 1000 1500
        do
            python wave_explicit_fusion.py --mesh-size $m --tile-size $ts
        done
    done
done
