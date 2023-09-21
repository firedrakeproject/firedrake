// gmsh -2 -format msh2 -clmax 0.001 sphere.geo
SetFactory("OpenCASCADE");
Sphere(1) = {0,0,0,0.1};
