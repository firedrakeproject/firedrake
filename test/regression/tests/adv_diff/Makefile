input: clean
	gmsh -2 -clscale 1.0 -o 6660.msh cdisk.geo
	gmsh -2 -clscale 2.0 -o 1552.msh cdisk.geo
	gmsh -2 -clscale 4.0 -o 376.msh  cdisk.geo
	gmsh -2 -clscale 8.0 -o 104.msh  cdisk.geo
	demo/meshes/gmsh2triangle --2d 6660.msh
	demo/meshes/gmsh2triangle --2d 1552.msh
	demo/meshes/gmsh2triangle --2d 376.msh
	demo/meshes/gmsh2triangle --2d 104.msh

.PHONY: clean input
clean:
	@rm -f *.out *.edge  *.ele  *.msh  *.node *.pyc
