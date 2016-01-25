LAUNCHERS=$FIREDRAKE_DIR/demos/tiling/launchers/cx1/pbs

if [ "$1" == "singlenode" ]
then
    echo "Executing single node experiments"
    qsub -l walltime=24:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
else
    echo "Executing multi node experiments..."
    for i in 1 2 4 8 12 16 20
    do
        echo "nodes="$i"..."
        qsub -l walltime=24:00:00 -l select=$i:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    done
fi
