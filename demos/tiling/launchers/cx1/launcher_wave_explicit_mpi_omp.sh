#!/bin/bash

#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=20:mem=16gb:icib=true
#PBS -q pqcdt

# Note: mpiexec.1pps runs hybrid mpi+openmp jobs
# Behind the scenes, it sets:
# I_MPI_PIN=yes
# I_MPI_PIN_MODE=lib
# I_MPI_PIN_DOMAIN=socket
# I_MPI_PIN_ORDER=compact
# KMP_AFFINITY=granularity=fine,compact,1,0

FIREDRAKE=$HOME/Projects/Firedrake/firedrake
TILING=$FIREDRAKE/demos/tiling
EXECUTABLE=$TILING/wave_explicit_fusion.py
MESHES=$TILING/meshes/wave_explicit

# Set SLOPE to OMP backend mode
export SLOPE_BACKEND=OMP


echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
cat /proc/cpuinfo | grep "model name" | uniq
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------
echo PBS: PYTHONPATH = $PYTHONPATH
echo ------------------------------------------------------
echo PBS: SLOPE_BACKEND = $SLOPE_BACKEND
echo ------------------------------------------------------


# Clean the remote cache, then dry runs on tiny mesh to generate kernels
$FIREDRAKE/scripts/firedrake-clean
for m in 10
do
    for ts in 4
    do
        mpiexec.1pps python $EXECUTABLE --mesh-size $m --tile-size $ts --num-unroll 0
        mpiexec.1pps python $EXECUTABLE --mesh-size $m --tile-size $ts --num-unroll 1
        mpiexec.1pps python $EXECUTABLE --mesh-size $m --tile-size $ts --num-unroll 2
    done
done


# Run the tests
for m in $MESHES"/wave_tank_0.125.msh"
do
    for ts in 1500 2000 2500 3000
    do
        # Hybrid mpi-openmp mode
        mpiexec.1pps python $EXECUTABLE --mesh-file $m --tile-size $ts --num-unroll 0
        mpiexec.1pps python $EXECUTABLE --mesh-file $m --tile-size $ts --num-unroll 1
        mpiexec.1pps python $EXECUTABLE --mesh-file $m --tile-size $ts --num-unroll 2

        # Openmp mode
        python $EXECUTABLE --mesh-file $m --tile-size $ts --num-unroll 0
        python $EXECUTABLE --mesh-file $m --tile-size $ts --num-unroll 1
        python $EXECUTABLE --mesh-file $m --tile-size $ts --num-unroll 2
    done
done
