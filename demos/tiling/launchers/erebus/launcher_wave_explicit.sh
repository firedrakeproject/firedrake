#!/bin/bash

FIREDRAKE=$FIREDRAKE_DIR
TILING=$FIREDRAKE/demos/tiling
EXECUTABLE=$TILING/wave_explicit.py
MESHES=/tmp/meshes_tiling

# Clean the remote cache, then dry runs on tiny mesh to generate kernels
$FIREDRAKE/scripts/firedrake-clean
echo "Running small problems to populate the cache..."
for nu in 0 1 2 3 4
do
    for p in "chunk" "metis"
    do
        for m in $MESHES"/wave_tank_1.0.msh"
        do
            for ts in 100
            do
                # OMP backend:
                export SLOPE_BACKEND=OMP
                export OMP_NUM_THREADS=4
                python $EXECUTABLE --mesh-file $m --tile-size $ts --part-mode $p --num-unroll $nu > /dev/null

                # MPI backend:
                export SLOPE_BACKEND=SEQUENTIAL
                export OMP_NUM_THREADS=1
                mpiexec python $EXECUTABLE --mesh-file $m --tile-size $ts --part-mode $p --num-unroll $nu > /dev/null
            done
        done
    done
done
echo "DONE!"


# Run the tests
for nu in 0 1 2 3 4
do
    for p in "chunk" "metis"
    do
        for m in $MESHES"/wave_tank_0.070.msh"
        do
            for ts in 2000 3000 5000 10000 20000 50000
            do
                # OMP backend:
                export SLOPE_BACKEND=OMP
                export OMP_NUM_THREADS=4
                python $EXECUTABLE --mesh-file $m --tile-size $ts --part-mode $p --num-unroll $nu >> output.txt

                # MPI backend:
                export SLOPE_BACKEND=SEQUENTIAL
                export OMP_NUM_THREADS=1
                mpirun --bind-to-core -np 4 python $EXECUTABLE --mesh-file $m --tile-size $ts --part-mode $p --num-unroll $nu >> output.txt
            done
        done
    done
done
