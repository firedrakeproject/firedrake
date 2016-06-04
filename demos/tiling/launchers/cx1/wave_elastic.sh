LAUNCHERS=$FIREDRAKE_DIR/demos/tiling/launchers/cx1/pbs/wave_elastic/

if [ "$1" == "singlenode" ]
then
    echo "Executing single node experiments: Haswell (20 cores, pqcdt)"
    qsub -v polys=1 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=2 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=3 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=4 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    echo "Executing single node experiments: Haswell (16 cores, defaultqueue)"
    qsub -v polys=1 -l walltime=72:00:00 -l select=1:ncpus=16:mem=32gb:haswell=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=2 -l walltime=72:00:00 -l select=1:ncpus=16:mem=32gb:haswell=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=3 -l walltime=72:00:00 -l select=1:ncpus=16:mem=32gb:haswell=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=4 -l walltime=72:00:00 -l select=1:ncpus=16:mem=32gb:haswell=true $LAUNCHERS/wave_elastic.pbs
    echo "Executing single node experiments: Ivy Bridge (20 cores, defaultqueue)"
    qsub -v polys=1 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=2 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=3 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=4 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
else
    echo "Executing multi node experiments: Haswell (with {1, 2, 4, 8, 12, 16, 20}x20 cores, pqcdt)"
    for i in 1 2 4 8 12 16 20
    do
        echo "nodes="$i"..."
        qsub -l walltime=72:00:00 -l select=$i:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    done
fi
