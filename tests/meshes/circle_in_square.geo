// Gmsh project created on Tue Aug  2 17:26:31 2016
Point(1) = {-2, 2, 0, 1};
Point(2) = {-2, -2, 0, 1};
Point(3) = {2, -2, 0, 1};
Point(4) = {2, 2, 0, 1};

Point(5) = {0, 0, 0, 1};
Point(6) = {1, 0, 0, 1};
Point(7) = {-1, 0, 0, 1};
Point(8) = {0, 1, 0, 1};
Point(9) = {0, -1, 0, 1};
Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};
Circle(5) = {8, 5, 6};
Circle(6) = {6, 5, 9};
Circle(7) = {9, 5, 7};
Circle(8) = {7, 5, 8};
Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {8, 5, 6, 7};
Plane Surface(11) = {9, 10};
Plane Surface(12) = {10};
Physical Line("Square") = {1, 2, 3, 4};
Physical Line("Circle") = {8, 7, 6, 5};
Physical Surface("SquareWithoutCircleSurface") = {11};
Physical Surface("CircleSurface") = {12};
