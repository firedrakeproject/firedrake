#!/bin/bash

export OMP_NUM_THREADS=4
export KMP_AFFINITY=scatter

for b in "SEQUENTIAL" "OMP"
do
    export SLOPE_BACKEND=$b
    for m in "(300.0,150.0)" "(600.0,300.0)"
    do
        # Non-tiled tests
        python wave_elastic.py --output 20 --mesh-size $m --num-unroll 0
        # Tiled tests
        for ts in 100 200 300 400 500 600 700 800 1000 1500
        do
            python wave_elastic.py --output 20 --mesh-size $m --tile-size $ts
        done
    done
done
