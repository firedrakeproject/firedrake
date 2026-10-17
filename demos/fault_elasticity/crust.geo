// A 200 km by 100 km cross-section of the crust. Lengths are in km.
// It contains a horizontal magma sill and a blind thrust fault that dips
// at 30 degrees. Both structures end inside the domain.
SetFactory("OpenCASCADE");

Rectangle(1) = {-100, -100, 0, 200, 100};

// Sill: 6 km wide, 3 km below the surface.
Point(5) = {-33, -3, 0};
Point(6) = {-27, -3, 0};
Line(5) = {5, 6};

// Blind thrust: from 18 km depth up to 3 km depth.
Point(7) = {0, -18, 0};
Point(8) = {15 * Sqrt(3), -3, 0};
Line(6) = {7, 8};

Curve{5, 6} In Surface{1};

Physical Curve("bottom", 1) = {1};
Physical Curve("sides", 2) = {2, 4};
Physical Curve("surface", 3) = {3};
Physical Curve("sill", 4) = {5};
Physical Curve("fault", 5) = {6};
Physical Surface("crust", 1) = {1};

// Refine the mesh near the sill, the fault and the ground surface.
Field[1] = Distance;
Field[1].CurvesList = {5, 6};
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.3;
Field[2].SizeMax = 10;
Field[2].DistMin = 1;
Field[2].DistMax = 60;
Background Field = 2;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
