SetFactory("OpenCASCADE");
//+
Rectangle(1) = {0, 0, 0, 0.6, 0.5, 0};
MeshSize {:} = 0.05;

// Curve 1: bottom (y=0), Curve 2: right (x=0.6)
// Curve 3: top (y=0.5), Curve 4: left (x=0)

Periodic Curve {2} = {4} Translate {0.6, 0, 0};
Periodic Curve {3} = {1} Translate {0, 0.5, 0};

Physical Surface(1) = {1};
Physical Curve(1) = {1};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {4};
