Point (1) = {0, 0 , 0, 1.};
Point (2) = {1, 0, 0,  1.};
Line (1) = {1, 2};
Extrude {0,1,0} {
  Line{1};Layers{1};
}

// Volume number for whole domain.
Physical Surface (1) = {5};
// Left
Physical Line(1) = {3};
// Right
Physical Line(2) = {4};
// Bottom
Physical Line(3) = {1};
// Top
Physical Line(4) = {2};
