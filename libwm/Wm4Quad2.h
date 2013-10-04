// Wild Magic Source Code
// David Eberly
// http://www.geometrictools.com
// Copyright (c) 1998-2008
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.  The license is available for reading at
// either of the locations:
//     http://www.gnu.org/copyleft/lgpl.html
//     http://www.geometrictools.com/License/WildMagicLicense.pdf
//
// Version: 4.0.0 (2006/06/28)

#ifndef WM4QUAD2_H
#define WM4QUAD2_H

#include "Wm4FoundationLIB.h"
#include "Wm4Vector2.h"

namespace Wm4
{

template <class Real>
class Quad2
{
public:
    // The quad is represented as an array of four vertices, V0, V1,
    // V2 and V3.

    // construction
    Quad2 ();  // uninitialized
    Quad2 (const Vector2<Real>& rkV0, const Vector2<Real>& rkV1,
        const Vector2<Real>& rkV2, const Vector2<Real>& rkV3);
    Quad2 (const Vector2<Real> akV[4]);

    Vector2<Real> V[4];
};

#include "Wm4Quad2.inl"

typedef Quad2<float> Quad2f;
typedef Quad2<double> Quad2d;

}

#endif
