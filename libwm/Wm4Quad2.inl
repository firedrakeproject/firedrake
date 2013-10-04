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

//----------------------------------------------------------------------------
template <class Real>
Quad2<Real>::Quad2 ()
{
    // uninitialized
}
//----------------------------------------------------------------------------
template <class Real>
Quad2<Real>::Quad2 (const Vector2<Real>& rkV0,
    const Vector2<Real>& rkV1, const Vector2<Real>& rkV2,
    const Vector2<Real>& rkV3)
{
    V[0] = rkV0;
    V[1] = rkV1;
    V[2] = rkV2;
    V[3] = rkV3;
}
//----------------------------------------------------------------------------
template <class Real>
Quad2<Real>::Quad2 (const Vector2<Real> akV[4])
{
    for (int i = 0; i < 4; i++)
    {
        V[i] = akV[i];
    }
}
//----------------------------------------------------------------------------
