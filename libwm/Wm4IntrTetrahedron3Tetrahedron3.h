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
// Version: 4.0.1 (2007/05/06)

#ifndef WM4INTRTETRAHEDRON3TETRAHEDRON3_H
#define WM4INTRTETRAHEDRON3TETRAHEDRON3_H

#include "Wm4FoundationLIB.h"
#include "Wm4Intersector.h"
#include "Wm4Tetrahedron3.h"

namespace Wm4
{

template <class Real>
class WM4_FOUNDATION_ITEM IntrTetrahedron3Tetrahedron3
    : public Intersector<Real,Vector3<Real> >
{
public:
    IntrTetrahedron3Tetrahedron3 (const Tetrahedron3<Real>& rkTetrahedron0,
        const Tetrahedron3<Real>& rkTetrahedron1);

    // object access
    const Tetrahedron3<Real>& GetTetrahedron0 () const;
    const Tetrahedron3<Real>& GetTetrahedron1 () const;

    // static query
    virtual bool Find ();

    // information about the intersection set
    const std::vector<Tetrahedron3<Real> >& GetIntersection () const;

private:
    static void SplitAndDecompose (Tetrahedron3<Real> kTetra,
        const Plane3<Real>& rkPlane,
        std::vector<Tetrahedron3<Real> >& rkInside);

    // the objects to intersect
    const Tetrahedron3<Real>* m_pkTetrahedron0;
    const Tetrahedron3<Real>* m_pkTetrahedron1;

    std::vector<Tetrahedron3<Real> > m_kIntersection;
};

typedef IntrTetrahedron3Tetrahedron3<float> IntrTetrahedron3Tetrahedron3f;
typedef IntrTetrahedron3Tetrahedron3<double> IntrTetrahedron3Tetrahedron3d;

}

#include "Wm4IntrTetrahedron3Tetrahedron3.cpp"

#endif
