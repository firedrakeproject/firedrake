LAUNCHERS=$FIREDRAKE_DIR/demos/tiling/launchers/cx1/pbs/wave_elastic/

if [ "$1" == "singlenode" ]; then
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
elif [ "$1" == "meshes" ]; then
    declare -a spacing=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)
    echo "Executing single node experiments: Ivy Bridge (20 cores, defaultqueue)"
    for h in ${spacing[@]}
    do
        echo "Executing p=1 and h=$h"
        qsub -v polys=1,mesh=$h -l walltime=48:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
    done
elif [ "$1" == "highp" ]; then
    echo "Executing single node experiments: Haswell (20 cores, pqcdt)"
    echo "Executing p=3 and h=0.6"
    qsub -v polys=3,mesh=0.6 -l walltime=48:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
elif [ "$1" == "tiles" ]; then
    echo "Executing single node experiments: Ivy Bridge (20 cores, defaultqueue)"
    echo "Executing p=1, h=0.6, em=5, many tiles and all parts"
    qsub -v polys=1,mesh=0.6,fixmode=1 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic_tmp.pbs
else
    echo "Executing multi node experiments: Haswell (with {1, 2, 4, 8, 12, 16, 20}x20 cores, pqcdt)"
    for i in 1 2 4 8 12 16 20
    do
        echo "nodes="$i"..."
        qsub -l walltime=72:00:00 -l select=$i:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    done
fi
