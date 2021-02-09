# README

Things do not work as expected. While the mesh is being properly generated in serial, a forward solve fails to converge on one of the test meshes.
MeshSize > 0.1 fails for this domain. 
I can pass the constant nullspace and it seems to converge but it is not immediately obvious why this mesh leads to a singular system.

Solving an elasticity problem also exhibits a disconnected mesh error. 


## Note 
One of the work arounds requires modifying ufl. ufl vector elements do not accept element variants.


