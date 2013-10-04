*realformat "%16.6f"
*intformat "%7i"
*Set Cond 01 *elems *canrepeat
*Add Cond 02 *elems *canrepeat
*Add Cond 03 *elems *canrepeat
*Add Cond 04 *elems *canrepeat
*Add Cond 05 *elems *canrepeat
*Add Cond 06 *elems *canrepeat
*Add Cond 07 *elems *canrepeat
*Add Cond 08 *elems *canrepeat
*Add Cond 09 *elems *canrepeat
*Add Cond 10 *elems *canrepeat
*Add Cond 11 *elems *canrepeat
*Add Cond 12 *elems *canrepeat
*Add Cond 13 *elems *canrepeat
*Add Cond 14 *elems *canrepeat
*Add Cond 15 *elems *canrepeat
*Add Cond 16 *elems *canrepeat
*Add Cond 17 *elems *canrepeat
*Add Cond 18 *elems *canrepeat
*Add Cond 19 *elems *canrepeat
*Add Cond 20 *elems *canrepeat
*set var nsurfaces(int)=CondNumEntities(int)
$MeshFormat
2 0 8
$EndMeshFormat
$Nodes
*format "%i"
*npoin
*loop nodes
*format "%i%f %f %f"
*NodesNum *NodesCoord
*end nodes
$EndNodes
$Elements
*format "%i"
*operation(nsurfaces+nelem)
*loop elems *OnlyinCond
*format "%i %i %i %i %i %i"
*LoopVar 2 2 *Cond(1,int) *Cond(1,int) *GlobalNodes
*end elems
*Set Cond region_ids *elems
*loop elems
*operation(nsurfaces+loopvar) *ElemsType 2 1 0 *ElemsConec
*end elems
$EndElements

