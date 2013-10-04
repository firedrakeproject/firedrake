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

#ifndef WM4INTRQUAD2QUAD2_H
#define WM4INTRQUAD2QUAD2_H

#include "Wm4FoundationLIB.h"
#include "Wm4Intersector.h"
#include "Wm4Intersector1.h"
#include "Wm4Quad2.h"

namespace Wm4
{

template <class Real>
class WM4_FOUNDATION_ITEM IntrQuad2Quad2
    : public Intersector<Real,Vector2<Real> >
{
public:
    IntrQuad2Quad2 (const Quad2<Real>& rkQuad0,
        const Quad2<Real>& rkQuad1);

    // object access
    const Quad2<Real>& GetQuad0 () const;
    const Quad2<Real>& GetQuad1 () const;

    // static queries
    virtual bool Find ();

    // information about the intersection set
    int GetQuantity () const;
    const Vector2<Real>& GetPoint (int i) const;

private:
    static void ClipConvexPolygonAgainstLine (const Vector2<Real>& rkN,
        Real fC, int& riQuantity, Vector2<Real> akV[6]);

    // the objects to intersect
    const Quad2<Real>* m_pkQuad0;
    const Quad2<Real>* m_pkQuad1;

    // information about the intersection set
    int m_iQuantity;
    Vector2<Real> m_akPoint[8];
};

typedef IntrQuad2Quad2<float> IntrQuad2Quad2f;
typedef IntrQuad2Quad2<double> IntrQuad2Quad2d;

}

#include "Wm4IntrQuad2Quad2.cpp"

#endif
