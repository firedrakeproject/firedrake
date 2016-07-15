LAUNCHERS=$FIREDRAKE_DIR/demos/tiling/launchers/cx1/pbs/wave_elastic/

if [ "$1" == "singlenode" ]; then
    echo "Executing single node experiments: Haswell (20 cores, pqcdt)"
    qsub -v polys=1,part=0 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=2,part=0 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=3,part=0 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=4,part=0 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    echo "Executing single node experiments: Haswell (16 cores, defaultqueue)"
    qsub -v polys=1,part=0 -l walltime=72:00:00 -l select=1:ncpus=16:mem=32gb:haswell=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=2,part=0 -l walltime=72:00:00 -l select=1:ncpus=16:mem=32gb:haswell=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=3,part=0 -l walltime=72:00:00 -l select=1:ncpus=16:mem=32gb:haswell=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=4,part=0 -l walltime=72:00:00 -l select=1:ncpus=16:mem=32gb:haswell=true $LAUNCHERS/wave_elastic.pbs
    echo "Executing single node experiments: Ivy Bridge (20 cores, defaultqueue)"
    qsub -v polys=1,part=0 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=2,part=0 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=3,part=0 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
    qsub -v polys=4,part=0 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
elif [ "$1" == 'testsuite' ]; then
    echo "Executing the main test suite"
    for poly in 1 2 3 4; do
        for h in 0.6 0.8; do
            for part in 0 1; do
                echo "Scheduling <poly=$poly,h=$h,part=$part> on ***Ivy Bridge (20 cores, defaultqueue)***"
                qsub -v polys=$poly,mesh=$h,part=$part -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
                echo "Scheduling <poly=$poly,h=$h,part=$part> on ***Haswell (20 cores, pqcdt)***"
                qsub -v polys=$poly,mesh=$h,part=$part -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
            done
        done
    done
elif [ "$1" == "meshes" ]; then
    declare -a spacing=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)
    echo "Executing single node experiments: Ivy Bridge (20 cores, defaultqueue)"
    for h in ${spacing[@]}
    do
        echo "Scheduling p=1 and h=$h"
        qsub -v polys=1,mesh=$h,part=0 -l walltime=48:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
    done
elif [ "$1" == "populator" ]; then
    echo "Populating cache for multi-node experiments: Haswell (20 cores, pqcdt)"
    for poly in 1 2 3 4; do
        for h in 0.6 0.45 0.3 0.225 0.15 0.115; do
            qsub -v polys=$poly,mesh=$h -l walltime=48:00:00 -l select=1:ncpus=20:mem=60gb:ivyb=true $LAUNCHERS/wave_elastic_populator.pbs
        done
    done
elif [ "$1" == "multinode-haswell" ]; then
    echo "Running multi-node experiments: Haswell (Nx20 cores, pqcdt)"
    for poly in 1 2 3 4; do
        qsub -v polys=$poly,mesh=0.6,nodename=0 -l walltime=48:00:00 -l select=1:ncpus=20:mem=48gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic_multinode.pbs
        qsub -v polys=$poly,mesh=0.45,nodename=0 -l walltime=48:00:00 -l select=2:ncpus=20:mem=48gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic_multinode.pbs
        qsub -v polys=$poly,mesh=0.3,nodename=0 -l walltime=48:00:00 -l select=4:ncpus=20:mem=48gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic_multinode.pbs
        qsub -v polys=$poly,mesh=0.225,nodename=0 -l walltime=48:00:00 -l select=8:ncpus=20:mem=48gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic_multinode.pbs
        qsub -v polys=$poly,mesh=0.15,nodename=0 -l walltime=48:00:00 -l select=16:ncpus=20:mem=48gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic_multinode.pbs
        qsub -v polys=$poly,mesh=0.115,nodename=0 -l walltime=48:00:00 -l select=32:ncpus=20:mem=48gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic_multinode.pbs
    done
elif [ "$1" == "multinode-ivy" ]; then
    echo "Running multi-node experiments: Ivy Bridge (Nx20 cores, defaultqueue)"
    for poly in 1 2 3 4; do
        qsub -v polys=$poly,mesh=0.6,nodename=1 -l walltime=48:00:00 -l select=1:ncpus=20:mem=48gb:ivyb=true $LAUNCHERS/wave_elastic_multinode.pbs
        qsub -v polys=$poly,mesh=0.45,nodename=1 -l walltime=48:00:00 -l select=2:ncpus=20:mem=48gb:ivyb=true $LAUNCHERS/wave_elastic_multinode.pbs
        qsub -v polys=$poly,mesh=0.3,nodename=1 -l walltime=48:00:00 -l select=4:ncpus=20:mem=48gb:ivyb=true $LAUNCHERS/wave_elastic_multinode.pbs
        qsub -v polys=$poly,mesh=0.225,nodename=1 -l walltime=48:00:00 -l select=8:ncpus=20:mem=48gb:ivyb=true $LAUNCHERS/wave_elastic_multinode.pbs
        qsub -v polys=$poly,mesh=0.15,nodename=1 -l walltime=48:00:00 -l select=16:ncpus=20:mem=48gb:ivyb=true $LAUNCHERS/wave_elastic_multinode.pbs
        qsub -v polys=$poly,mesh=0.115,nodename=1 -l walltime=48:00:00 -l select=32:ncpus=20:mem=48gb:ivyb=true $LAUNCHERS/wave_elastic_multinode.pbs
    done
elif [ "$1" == "playwithmpi" ]; then
    echo "Executing single node experiments: Haswell (20 cores, using 10 processes in a single NUMA domain, pqcdt)"
    echo "Executing p=2 and h=0.9"
    qsub -v polys=2,mesh=0.9,part=0,nprocs=10 -l walltime=48:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    echo "Executing single node experiments: Haswell (20 cores, using 20 processes in scatter mode, pqcdt)"
    echo "Executing p=2 and h=0.8"
    qsub -v polys=2,mesh=0.8,part=0,nprocs=20 -l walltime=48:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
elif [ "$1" == "tiles" ]; then
    echo "Executing single node experiments: Ivy Bridge (20 cores, defaultqueue)"
    qsub -v polys=4,mesh=0.6,part=0,fixmode=1 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true $LAUNCHERS/wave_elastic.pbs
elif [ "$1" == "onlylog" ]; then
    echo "Executing single node experiments for collecting tiling summaries: Haswell (20 cores, pqcdt)"
    for poly in 1 2 3 4; do
        echo "Scheduling <poly=$poly,h=0.8,part=0> on ***Haswell (20 cores, pqcdt)***"
        qsub -v polys=$poly,mesh=0.8,part=0,onlylog=1 -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    done
else
    echo "Executing multi node experiments: Haswell (with {1, 2, 4, 8, 12, 16, 20}x20 cores, pqcdt)"
    for i in 1 2 4 8 12 16 20
    do
        echo "nodes="$i"..."
        qsub -v part=0 -l walltime=72:00:00 -l select=$i:ncpus=20:mem=32gb:icib=true -q pqcdt $LAUNCHERS/wave_elastic.pbs
    done
fi
