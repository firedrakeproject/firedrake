LAUNCHERS=$FIREDRAKE_DIR/demos/tiling/launchers/cx1/pbs

echo "Execute STREAM on Haswell (20 cores)"
qsub -v NTHREADS=20 -l walltime=12:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/stream.pbs

echo "Execute STREAM on Ivy Bridge (20 cores)"
qsub -v NTHREADS=20 -l walltime=12:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/stream.pbs
