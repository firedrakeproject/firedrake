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

#ifndef WM4TETRAHEDRON3_H
#define WM4TETRAHEDRON3_H

#include "Wm4FoundationLIB.h"
#include "Wm4Plane3.h"

namespace Wm4
{

template <class Real>
class Tetrahedron3
{
public:
    // The vertices are ordered so that the triangular faces are
    // counterclockwise when viewed by an observer outside the tetrahedron:
    //   face 0 = <V[0],V[2],V[1]>
    //   face 1 = <V[0],V[1],V[3]>
    //   face 2 = <V[0],V[3],V[2]>
    //   face 3 = <V[1],V[2],V[3]>

    // Construction.
    Tetrahedron3 ();  // uninitialized
    Tetrahedron3 (const Vector3<Real>& rkV0, const Vector3<Real>& rkV1,
        const Vector3<Real>& rkV2, const Vector3<Real>& rkV3);
    Tetrahedron3 (const Vector3<Real> akV[4]);

    // Get the vertex indices for the specified face.
    void GetFaceIndices (int iFace, int aiIndex[3]) const;

    // Construct the planes of the faces.  The planes have outer pointing
    // normal vectors.  The normals may be specified to be unit length.  The
    // plane indexing is the same as the face indexing mentioned previously.
    void GetPlanes (Plane3<Real> akPlane[4], bool bUnitLengthNormals) const;

    Real GetVolume() const;

    Vector3<Real> V[4];
};

#include "Wm4Tetrahedron3.inl"

typedef Tetrahedron3<float> Tetrahedron3f;
typedef Tetrahedron3<double> Tetrahedron3d;

}

#endif
