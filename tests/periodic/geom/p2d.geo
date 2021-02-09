SetFactory("OpenCASCADE");
//+
Rectangle(1) = {0, 0, 0, 0.6, 0.5, 0};
MeshSize {:} = 0.1;
MeshSize {:} = 0.1;

// S2: Right Side
// S4: Left Side

Periodic Curve {2} = {4} Translate {0.6,0,0};

Physical Surface(1) = {1};
Physical Curve(1) = {1};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {4};
