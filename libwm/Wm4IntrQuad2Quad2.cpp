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
#include "Wm4IntrQuad2Quad2.h"

namespace Wm4
{
//----------------------------------------------------------------------------
template <class Real>
IntrQuad2Quad2<Real>::IntrQuad2Quad2 (
    const Quad2<Real>& rkQuad0, const Quad2<Real>& rkQuad1)
    :
    m_pkQuad0(&rkQuad0),
    m_pkQuad1(&rkQuad1)
{
    m_iQuantity = 0;
}
//----------------------------------------------------------------------------
template <class Real>
const Quad2<Real>& IntrQuad2Quad2<Real>::GetQuad0 () const
{
    return *m_pkQuad0;
}
//----------------------------------------------------------------------------
template <class Real>
const Quad2<Real>& IntrQuad2Quad2<Real>::GetQuad1 () const
{
    return *m_pkQuad1;
}
//----------------------------------------------------------------------------
template <class Real>
bool IntrQuad2Quad2<Real>::Find ()
{
    // The potential intersection is initialized to quad1.  The set of
    // vertices is refined based on clipping against each edge of quad0.
    m_iQuantity = 4;
    for (int i = 0; i < 4; i++)
    {
        m_akPoint[i] = m_pkQuad1->V[i];
    }

    for (int i1 = 3, i0 = 0; i0 < 4; i1 = i0, i0++)
    {
        // clip against edge <V0[i1],V0[i0]>
        Vector2<Real> kN(
            m_pkQuad0->V[i1].Y() - m_pkQuad0->V[i0].Y(),
            m_pkQuad0->V[i0].X() - m_pkQuad0->V[i1].X());
        Real fC = kN.Dot(m_pkQuad0->V[i1]);
        ClipConvexPolygonAgainstLine(kN,fC,m_iQuantity,m_akPoint);
        if (m_iQuantity == 0)
        {
            // quad completely clipped, no intersection occurs
            return false;
        }
    }

    return true;
}
//----------------------------------------------------------------------------
template <class Real>
int IntrQuad2Quad2<Real>::GetQuantity () const
{
    return m_iQuantity;
}
//----------------------------------------------------------------------------
template <class Real>
const Vector2<Real>& IntrQuad2Quad2<Real>::GetPoint (int i) const
{
    assert(0 <= i && i < m_iQuantity);
    return m_akPoint[i];
}
//----------------------------------------------------------------------------
template <class Real>
void IntrQuad2Quad2<Real>::ClipConvexPolygonAgainstLine (
    const Vector2<Real>& rkN, Real fC, int& riQuantity,
    Vector2<Real> akV[8])
{
    // The input vertices are assumed to be in counterclockwise order.  The
    // ordering is an invariant of this function.

    // test on which side of line the vertices are
    int iPositive = 0, iNegative = 0, iPIndex = -1;
    Real afTest[8];
    int i;
    for (i = 0; i < riQuantity; i++)
    {
        afTest[i] = rkN.Dot(akV[i]) - fC;
        if (afTest[i] > (Real)0.0)
        {
            iPositive++;
            if (iPIndex < 0)
            {
                iPIndex = i;
            }
        }
        else if (afTest[i] < (Real)0.0)
        {
            iNegative++;
        }
    }

    if (iPositive > 0)
    {
        if (iNegative > 0)
        {
            // line transversely intersects polygon
            Vector2<Real> akCV[8];
            int iCQuantity = 0, iCur, iPrv;
            Real fT;

            if (iPIndex > 0)
            {
                // first clip vertex on line
                iCur = iPIndex;
                iPrv = iCur-1;
                fT = afTest[iCur]/(afTest[iCur] - afTest[iPrv]);
                akCV[iCQuantity++] = akV[iCur]+fT*(akV[iPrv]-akV[iCur]);

                // vertices on positive side of line
                while (iCur < riQuantity && afTest[iCur] > (Real)0.0)
                {
                    akCV[iCQuantity++] = akV[iCur++];
                }

                // last clip vertex on line
                if (iCur < riQuantity)
                {
                    iPrv = iCur-1;
                }
                else
                {
                    iCur = 0;
                    iPrv = riQuantity - 1;
                }
                fT = afTest[iCur]/(afTest[iCur] - afTest[iPrv]);
                akCV[iCQuantity++] = akV[iCur]+fT*(akV[iPrv]-akV[iCur]);
            }
            else  // iPIndex is 0
            {
                // vertices on positive side of line
                iCur = 0;
                while (iCur < riQuantity && afTest[iCur] > (Real)0.0)
                {
                    akCV[iCQuantity++] = akV[iCur++];
                }

                // last clip vertex on line
                iPrv = iCur-1;
                fT = afTest[iCur]/(afTest[iCur] - afTest[iPrv]);
                akCV[iCQuantity++] = akV[iCur]+fT*(akV[iPrv]-akV[iCur]);

                // skip vertices on negative side
                while (iCur < riQuantity && afTest[iCur] <= (Real)0.0)
                {
                    iCur++;
                }

                // first clip vertex on line
                if (iCur < riQuantity)
                {
                    iPrv = iCur-1;
                    fT = afTest[iCur]/(afTest[iCur] - afTest[iPrv]);
                    akCV[iCQuantity++] = akV[iCur]+fT*(akV[iPrv]-akV[iCur]);

                    // vertices on positive side of line
                    while (iCur < riQuantity && afTest[iCur] > (Real)0.0)
                    {
                        akCV[iCQuantity++] = akV[iCur++];
                    }
                }
                else
                {
                    // iCur = 0
                    iPrv = riQuantity - 1;
                    fT = afTest[0]/(afTest[0] - afTest[iPrv]);
                    akCV[iCQuantity++] = akV[0]+fT*(akV[iPrv]-akV[0]);
                }
            }

            riQuantity = iCQuantity;
            size_t uiSize = iCQuantity*sizeof(Vector2<Real>);
            System::Memcpy(akV,uiSize,akCV,uiSize);
        }
        // else polygon fully on positive side of line, nothing to do
    }
    else
    {
        // polygon does not intersect positive side of line, clip all
        riQuantity = 0;
    }
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// explicit instantiation
//----------------------------------------------------------------------------
template WM4_FOUNDATION_ITEM
class IntrQuad2Quad2<float>;

template WM4_FOUNDATION_ITEM
class IntrQuad2Quad2<double>;
//----------------------------------------------------------------------------
}
