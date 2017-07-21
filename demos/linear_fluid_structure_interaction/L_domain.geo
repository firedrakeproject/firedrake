x0 = 0;
xBf = 2.;
z0 = 0;
zBf = 20.;
xWf = -20.;
zWf = 10.;
lc = 0.01;

// nondim units
L = Fabs(zWf - z0);
xBf /= L;
zBf /= L;
xWf /= L;
zWf /= L;


NxW = 8; // # elements in x dir in water
NzW = 4; // # elements in z dir in water
NxB = 2; // # elements in x dir in beam
NzB = 8; // # elements in z dir in beam

Point(1) = {x0, z0, 0, lc};
Point(2) = {xBf, z0, 0, lc};
Point(3) = {xBf, zWf, 0, lc};
Point(4) = {xBf, zBf, 0, lc};
Point(5) = {x0, zBf, 0, lc};
Point(6) = {x0, zWf, 0, lc};
Point(7) = {xWf, zWf, 0, lc};
Point(8) = {xWf, z0, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Line(9) = {6, 1};
Line(10) = {3, 6};


Line Loop(11) = {7, 8, -9, 6};
Plane Surface(12) = {11};
Line Loop(13) = {9, 1, 2, 10};
Plane Surface(14) = {13};
Line Loop(15) = {5, -10, 3, 4};
Plane Surface(16) = {15};

Transfinite Line{1, 4, 10} = NxB + 1;
Transfinite Line{2, 7, 9} = NzW + 1;
Transfinite Line{3, 5} = NzB - NzW + 1;
Transfinite Line{6, 8} = NxW + 1;

Transfinite Surface{12, 14, 16};

Physical Line(1) = {1};
Physical Line(4) = {4};
Physical Line(5) = {5};
Physical Line(6) = {6};
Physical Line(7) = {7};
Physical Line(8) = {8};
Physical Line(9) = {9};
Physical Line(10) = {2,3};

Physical Surface(1) = {12};
Physical Surface(2) = {14, 16};

lc = DefineNumber[ 0.1, Name "Parameters/lc" ];
