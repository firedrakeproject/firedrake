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

#include "Wm4FoundationPCH.h"
#include "Wm4IntrTetrahedron3Tetrahedron3.h"

namespace Wm4
{
//----------------------------------------------------------------------------
template <class Real>
IntrTetrahedron3Tetrahedron3<Real>::IntrTetrahedron3Tetrahedron3 (
    const Tetrahedron3<Real>& rkTetrahedron0,
    const Tetrahedron3<Real>& rkTetrahedron1)
    :
    m_pkTetrahedron0(&rkTetrahedron0),
    m_pkTetrahedron1(&rkTetrahedron1)
{
}
//----------------------------------------------------------------------------
template <class Real>
const Tetrahedron3<Real>&
IntrTetrahedron3Tetrahedron3<Real>::GetTetrahedron0 () const
{
    return *m_pkTetrahedron0;
}
//----------------------------------------------------------------------------
template <class Real>
const Tetrahedron3<Real>&
IntrTetrahedron3Tetrahedron3<Real>::GetTetrahedron1 () const
{
    return *m_pkTetrahedron1;
}
//----------------------------------------------------------------------------
template <class Real>
bool IntrTetrahedron3Tetrahedron3<Real>::Find ()
{
    // build planar faces of tetrahedron0
    Plane3<Real> akPlane[4];
    m_pkTetrahedron0->GetPlanes(akPlane,true);

    // initial object to clip is tetrahedron1
    m_kIntersection.clear();
    m_kIntersection.push_back(*m_pkTetrahedron1);

    // clip tetrahedron1 against planes of tetrahedron0
    for (int iP = 0; iP < 4; iP++)
    {
        std::vector<Tetrahedron3<Real> > kInside;
        for (int iT = 0; iT < (int)m_kIntersection.size(); iT++)
        {
            SplitAndDecompose(m_kIntersection[iT],akPlane[iP],kInside);
        }
        m_kIntersection = kInside;
    }

    return m_kIntersection.size() > 0;
}
//----------------------------------------------------------------------------
template <class Real>
const std::vector<Tetrahedron3<Real> >&
IntrTetrahedron3Tetrahedron3<Real>::GetIntersection () const
{
    return m_kIntersection;
}
//----------------------------------------------------------------------------
template <class Real>
void IntrTetrahedron3Tetrahedron3<Real>::SplitAndDecompose (
    Tetrahedron3<Real> kTetra, const Plane3<Real>& rkPlane,
    std::vector<Tetrahedron3<Real> >& rkInside)
{
    // determine on which side of the plane the points of the tetrahedron lie
    Real afC[4];
    int i, aiP[4], aiN[4], aiZ[4];
    int iPositive = 0, iNegative = 0, iZero = 0;

    for (i = 0; i < 4; i++)
    {
        afC[i] = rkPlane.DistanceTo(kTetra.V[i]);
        if (afC[i] > (Real)0.0)
        {
            aiP[iPositive++] = i;
        }
        else if (afC[i] < (Real)0.0)
        {
            aiN[iNegative++] = i;
        }
        else
        {
            aiZ[iZero++] = i;
        }
    }

    // For a split to occur, one of the c_i must be positive and one must
    // be negative.

    if (iNegative == 0)
    {
        // tetrahedron is completely on the positive side of plane, full clip
        return;
    }

    if (iPositive == 0)
    {
        // tetrahedron is completely on the negative side of plane
        rkInside.push_back(kTetra);
        return;
    }

    // Tetrahedron is split by plane.  Determine how it is split and how to
    // decompose the negative-side portion into tetrahedra (6 cases).
    Real fW0, fW1, fInvCDiff;
    Vector3<Real> akIntp[4];

    if (iPositive == 3)
    {
        // +++-
        for (i = 0; i < iPositive; i++)
        {
            fInvCDiff = ((Real)1.0)/(afC[aiP[i]] - afC[aiN[0]]);
            fW0 = -afC[aiN[0]]*fInvCDiff;
            fW1 = +afC[aiP[i]]*fInvCDiff;
            kTetra.V[aiP[i]] = fW0*kTetra.V[aiP[i]] + fW1*kTetra.V[aiN[0]];
        }
        rkInside.push_back(kTetra);
    }
    else if (iPositive == 2)
    {
        if (iNegative == 2)
        {
            // ++--
            for (i = 0; i < iPositive; i++)
            {
                fInvCDiff = ((Real)1.0)/(afC[aiP[i]]-afC[aiN[0]]);
                fW0 = -afC[aiN[0]]*fInvCDiff;
                fW1 = +afC[aiP[i]]*fInvCDiff;
                akIntp[i] = fW0*kTetra.V[aiP[i]] + fW1*kTetra.V[aiN[0]];
            }
            for (i = 0; i < iNegative; i++)
            {
                fInvCDiff = ((Real)1.0)/(afC[aiP[i]]-afC[aiN[1]]);
                fW0 = -afC[aiN[1]]*fInvCDiff;
                fW1 = +afC[aiP[i]]*fInvCDiff;
                akIntp[i+2] = fW0*kTetra.V[aiP[i]] + fW1*kTetra.V[aiN[1]];
            }

            kTetra.V[aiP[0]] = akIntp[2];
            kTetra.V[aiP[1]] = akIntp[1];
            rkInside.push_back(kTetra);

            rkInside.push_back(Tetrahedron3<Real>(kTetra.V[aiN[1]],akIntp[3],
                akIntp[2],akIntp[1]));

            rkInside.push_back(Tetrahedron3<Real>(kTetra.V[aiN[0]],akIntp[0],
                akIntp[1],akIntp[2]));
        }
        else
        {
            // ++-0
            for (i = 0; i < iPositive; i++)
            {
                fInvCDiff = ((Real)1.0)/(afC[aiP[i]]-afC[aiN[0]]);
                fW0 = -afC[aiN[0]]*fInvCDiff;
                fW1 = +afC[aiP[i]]*fInvCDiff;
                kTetra.V[aiP[i]] = fW0*kTetra.V[aiP[i]] +
                    fW1*kTetra.V[aiN[0]];
            }
            rkInside.push_back(kTetra);
        }
    }
    else if (iPositive == 1)
    {
        if (iNegative == 3)
        {
            // +---
            for (i = 0; i < iNegative; i++)
            {
                fInvCDiff = ((Real)1.0)/(afC[aiP[0]]-afC[aiN[i]]);
                fW0 = -afC[aiN[i]]*fInvCDiff;
                fW1 = +afC[aiP[0]]*fInvCDiff;
                akIntp[i] = fW0*kTetra.V[aiP[0]] + fW1*kTetra.V[aiN[i]];
            }

            kTetra.V[aiP[0]] = akIntp[0];
            rkInside.push_back(kTetra);

            rkInside.push_back(Tetrahedron3<Real>(akIntp[0],kTetra.V[aiN[1]],
                kTetra.V[aiN[2]],akIntp[1]));

            rkInside.push_back(Tetrahedron3<Real>(kTetra.V[aiN[2]],akIntp[1],
                akIntp[2],akIntp[0]));
        }
        else if (iNegative == 2)
        {
            // +--0
            for (i = 0; i < iNegative; i++)
            {
                fInvCDiff = ((Real)1.0)/(afC[aiP[0]]-afC[aiN[i]]);
                fW0 = -afC[aiN[i]]*fInvCDiff;
                fW1 = +afC[aiP[0]]*fInvCDiff;
                akIntp[i] = fW0*kTetra.V[aiP[0]] + fW1*kTetra.V[aiN[i]];
            }

            kTetra.V[aiP[0]] = akIntp[0];
            rkInside.push_back(kTetra);

            rkInside.push_back(Tetrahedron3<Real>(akIntp[1],kTetra.V[aiZ[0]],
                kTetra.V[aiN[1]],akIntp[0]));
        }
        else
        {
            // +-00
            fInvCDiff = ((Real)1.0)/(afC[aiP[0]]-afC[aiN[0]]);
            fW0 = -afC[aiN[0]]*fInvCDiff;
            fW1 = +afC[aiP[0]]*fInvCDiff;
            kTetra.V[aiP[0]] = fW0*kTetra.V[aiP[0]] + fW1*kTetra.V[aiN[0]];
            rkInside.push_back(kTetra);
        }
    }
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// explicit instantiation
//----------------------------------------------------------------------------
template WM4_FOUNDATION_ITEM
class IntrTetrahedron3Tetrahedron3<float>;

template WM4_FOUNDATION_ITEM
class IntrTetrahedron3Tetrahedron3<double>;

template WM4_FOUNDATION_ITEM
class IntrTetrahedron3Tetrahedron3<long double>;
//----------------------------------------------------------------------------
}
