#!/bin/sh

export OMP_NUM_THREADS=4
export KMP_AFFINITY=scatter

for b in "SEQUENTIAL" "OMP"
do
    for m in "(300.0,150.0,2.5)" "(600.0,300.0,2.5)"
    do
        for ts in 100 200 300 400 500 600 700 800 1000 1500
        do
            export SLOPE_BACKEND=$b
            python wave_elastic.py --output 20 --mesh-size $m --tile-size $ts
        done
    done
done
