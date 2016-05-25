#!/bin/bash

rm -rf /data/output/

OPTS="--output 1000 --flatten True --nocache True"
TILE_OPTS="--extra-halo 1 --fusion-mode only_tile --glb-maps True --coloring default"

export OMP_NUM_THREADS=1
export SLOPE_BACKEND=SEQUENTIAL

# Tile sizes for each poly order
declare -a ts_p1=(150 230 310 400)
declare -a ts_p2=(70 140 200 300)
declare -a ts_p3=(40 70 100)
declare -a ts_p4=(30 60 90)

# Partition modes for each poly order
declare -a part_p1=("chunk")
declare -a part_p2=("chunk")
declare -a part_p3=("chunk" "metis")
declare -a part_p4=("metis")

for poly in 1 2 3 4
do
    output_file="output_p"$poly".txt"
    rm -f $output_file
    touch $output_file
    echo "Polynomial order "$poly
    for MESH in "--mesh-size (300.0,150.0,0.8)" "--mesh-file /data/meshes/domain_h08.msh --h 0.8"
    do
        echo "    Running "$MESH
        echo "        Untiled ..."
        PROGRAM="python wave_elastic.py --poly-order $poly $MESH $OPTS --num-unroll 0"
        mpirun -np 4 --bind-to-core python wave_elastic.py --poly-order $poly $MESH $OPTS --num-unroll 0 1>> $output_file 2>> $output_file
        mpirun -np 4 --bind-to-core python wave_elastic.py --poly-order $poly $MESH $OPTS --num-unroll 0 1>> $output_file 2>> $output_file
        part_p="part_p$poly[*]"
        for p in ${!part_p}
        do
            for em in 2 3 4 5
            do
                ts_p="ts_p$poly[*]"
                for ts in ${!ts_p}
                do
                    echo "        Tiled (pm="$p", ts="$ts", em="$em") ..."
                    mpirun -np 4 --bind-to-core python wave_elastic.py --poly-order $poly $MESH $OPTS --num-unroll 1 --tile-size $ts --part-mode $p --explicit-mode $em $TILE_OPT 1>> $output_file 2>> $output_file
                done
            done
        done
    done
done

export OMP_NUM_THREADS=4
export KMP_AFFINITY=scatter
export SLOPE_BACKEND=OMP
echo "No OMP experiments set"
