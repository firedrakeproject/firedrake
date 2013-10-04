#!/bin/bash

surfaces="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20"
basename=problem_type-fluidity

### You should only need to change the stuff above this line

mkdir -p ${basename}.gid
cnd=${basename}.gid/${basename}.cnd
bas=${basename}.gid/${basename}.bas

nsurfaces=0
for i in $surfaces
do
    ((nsurfaces=nsurfaces+1))
done

# Write the cnd file
id=0
rm -f $cnd
for i in $surfaces
do
((id=id+1))
    cat >> $cnd <<EOF
NUMBER: $id CONDITION:$i
CONDTYPE: over surfaces
CONDMESHTYPE: over face elements
CANREPEAT: YES
QUESTION: SET?(No_value_required):
VALUE:0
END CONDITION
EOF
done

# Write the bas file
cat > $bas <<EOF
*realformat "%16.6f"
*intformat "%7i"
EOF

for i in $surfaces
do
upper=`echo $i | tr [a-z] [A-Z]`
cat >> $bas <<EOF
*Set Cond $i *elems *Canrepeat
*set var $upper(int)=CondNumEntities(int)
EOF
done

cat >> $bas <<EOF
Nodes   Elems $surfaces
EOF
counts="*npoin  *nelem $nsurfaces "
for i in $surfaces
do
upper=`echo $i | tr [a-z] [A-Z]`
counts="$counts *$upper"
done

cat >> $bas <<EOF
$counts
node      x       y      z
*loop nodes
*NodesNum *NodesCoord
*end nodes
Connectivity
*loop elems
*ElemsNnode *ElemsConec
*end elems
EOF

for i in $surfaces
do
upper=`echo $i | tr [a-z] [A-Z]`
cat >> $bas <<EOF
Face Element nodes for $i
*Set Cond $i *elems
*loop elems *OnlyinCond
*GlobalNodes
*end elems
EOF
done
