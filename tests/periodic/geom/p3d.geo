SetFactory("OpenCASCADE");

Box(1) = {0,0,0,1,1,1};
MeshSize {:} = 0.1;
MeshSize {1} = 0.1;

// Periodic Surface
// S2: Right side
// S1: Left Side ( Translate the 1 surface by 1)
Periodic Surface {2} = {1} Translate {1,0,0};

Physical Volume(1) = {1};
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};
Physical Surface(4) = {4};
Physical Surface(5) = {5};
Physical Surface(6) = {6};
