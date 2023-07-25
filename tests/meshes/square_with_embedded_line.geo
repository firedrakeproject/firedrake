//
// Labels on lines are the mesh markers
//               4
//
//       6-------5-------4
//       |       |       |
//       |       |       |
//       |       |       |
//     1 |       5       | 2
//       |       |       |
//       |       |       |
//       |       |       |
//       1-------2-------3
//
//               3


Point(1) = {0, 0, 0, 0.1};
Point(2) = {0.5, 0, 0, 0.1};
Point(3) = {1, 0, 0, 0.1};
Point(4) = {1, 1, 0, 0.1};
Point(5) = {0.5, 1, 0, 0.1};
Point(6) = {0, 1, 0, 0.1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Line Loop(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};

Line(7) = {2, 5};
Line{7} In Surface{1};

Physical Surface(1) = {1};

Physical Line(1) = {6};
Physical Line(2) = {3};
Physical Line(3) = {1, 2};
Physical Line(4) = {4, 5};
Physical Line(5) = {7};
