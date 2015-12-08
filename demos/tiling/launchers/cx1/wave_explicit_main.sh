LAUNCHERS=$FIREDRAKE_DIR/demos/tiling/launchers/cx1

qsub -v NNODES=1 $LAUNCHERS/launcher_wave_explicit.sh
qsub -v NNODES=2 $LAUNCHERS/launcher_wave_explicit.sh
qsub -v NNODES=4 $LAUNCHERS/launcher_wave_explicit.sh
qsub -v NNODES=8 $LAUNCHERS/launcher_wave_explicit.sh
qsub -v NNODES=12 $LAUNCHERS/launcher_wave_explicit.sh
qsub -v NNODES=16 $LAUNCHERS/launcher_wave_explicit.sh
qsub -v NNODES=20 $LAUNCHERS/launcher_wave_explicit.sh
