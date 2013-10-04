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
// Version: 4.0.2 (2008/09/15)

#include "Wm4FoundationPCH.h"
#include "Wm4IntrBox3Box3.h"
#include "Wm4IntrUtility3.h"

namespace Wm4
{
//----------------------------------------------------------------------------
template <class Real>
IntrBox3Box3<Real>::IntrBox3Box3 (const Box3<Real>& rkBox0,
    const Box3<Real>& rkBox1)
    :
    m_pkBox0(&rkBox0),
    m_pkBox1(&rkBox1)
{
    m_iQuantity = 0;
}
//----------------------------------------------------------------------------
template <class Real>
const Box3<Real>& IntrBox3Box3<Real>::GetBox0 () const
{
    return *m_pkBox0;
}
//----------------------------------------------------------------------------
template <class Real>
const Box3<Real>& IntrBox3Box3<Real>::GetBox1 () const
{
    return *m_pkBox1;
}
//----------------------------------------------------------------------------
template <class Real>
bool IntrBox3Box3<Real>::Test ()
{
    // Cutoff for cosine of angles between box axes.  This is used to catch
    // the cases when at least one pair of axes are parallel.  If this
    // happens, there is no need to test for separation along the
    // Cross(A[i],B[j]) directions.
    const Real fCutoff = (Real)1.0 - Math<Real>::ZERO_TOLERANCE;
    bool bExistsParallelPair = false;
    int i;

    // convenience variables
    const Vector3<Real>* akA = m_pkBox0->Axis;
    const Vector3<Real>* akB = m_pkBox1->Axis;
    const Real* afEA = m_pkBox0->Extent;
    const Real* afEB = m_pkBox1->Extent;

    // compute difference of box centers, D = C1-C0
    Vector3<Real> kD = m_pkBox1->Center - m_pkBox0->Center;

    Real aafC[3][3];     // matrix C = A^T B, c_{ij} = Dot(A_i,B_j)
    Real aafAbsC[3][3];  // |c_{ij}|
    Real afAD[3];        // Dot(A_i,D)
    Real fR0, fR1, fR;   // interval radii and distance between centers
    Real fR01;           // = R0 + R1

    // axis C0+t*A0
    for (i = 0; i < 3; i++)
    {
        aafC[0][i] = akA[0].Dot(akB[i]);
        aafAbsC[0][i] = Math<Real>::FAbs(aafC[0][i]);
        if (aafAbsC[0][i] > fCutoff)
        {
            bExistsParallelPair = true;
        }
    }
    afAD[0] = akA[0].Dot(kD);
    fR = Math<Real>::FAbs(afAD[0]);
    fR1 = afEB[0]*aafAbsC[0][0]+afEB[1]*aafAbsC[0][1]+afEB[2]*aafAbsC[0][2];
    fR01 = afEA[0] + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*A1
    for (i = 0; i < 3; i++)
    {
        aafC[1][i] = akA[1].Dot(akB[i]);
        aafAbsC[1][i] = Math<Real>::FAbs(aafC[1][i]);
        if (aafAbsC[1][i] > fCutoff)
        {
            bExistsParallelPair = true;
        }
    }
    afAD[1] = akA[1].Dot(kD);
    fR = Math<Real>::FAbs(afAD[1]);
    fR1 = afEB[0]*aafAbsC[1][0]+afEB[1]*aafAbsC[1][1]+afEB[2]*aafAbsC[1][2];
    fR01 = afEA[1] + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*A2
    for (i = 0; i < 3; i++)
    {
        aafC[2][i] = akA[2].Dot(akB[i]);
        aafAbsC[2][i] = Math<Real>::FAbs(aafC[2][i]);
        if (aafAbsC[2][i] > fCutoff)
        {
            bExistsParallelPair = true;
        }
    }
    afAD[2] = akA[2].Dot(kD);
    fR = Math<Real>::FAbs(afAD[2]);
    fR1 = afEB[0]*aafAbsC[2][0]+afEB[1]*aafAbsC[2][1]+afEB[2]*aafAbsC[2][2];
    fR01 = afEA[2] + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*B0
    fR = Math<Real>::FAbs(akB[0].Dot(kD));
    fR0 = afEA[0]*aafAbsC[0][0]+afEA[1]*aafAbsC[1][0]+afEA[2]*aafAbsC[2][0];
    fR01 = fR0 + afEB[0];
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*B1
    fR = Math<Real>::FAbs(akB[1].Dot(kD));
    fR0 = afEA[0]*aafAbsC[0][1]+afEA[1]*aafAbsC[1][1]+afEA[2]*aafAbsC[2][1];
    fR01 = fR0 + afEB[1];
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*B2
    fR = Math<Real>::FAbs(akB[2].Dot(kD));
    fR0 = afEA[0]*aafAbsC[0][2]+afEA[1]*aafAbsC[1][2]+afEA[2]*aafAbsC[2][2];
    fR01 = fR0 + afEB[2];
    if (fR > fR01)
    {
        return false;
    }

    // At least one pair of box axes was parallel, so the separation is
    // effectively in 2D where checking the "edge" normals is sufficient for
    // the separation of the boxes.
    if (bExistsParallelPair)
    {
        return true;
    }

    // axis C0+t*A0xB0
    fR = Math<Real>::FAbs(afAD[2]*aafC[1][0]-afAD[1]*aafC[2][0]);
    fR0 = afEA[1]*aafAbsC[2][0] + afEA[2]*aafAbsC[1][0];
    fR1 = afEB[1]*aafAbsC[0][2] + afEB[2]*aafAbsC[0][1];
    fR01 = fR0 + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*A0xB1
    fR = Math<Real>::FAbs(afAD[2]*aafC[1][1]-afAD[1]*aafC[2][1]);
    fR0 = afEA[1]*aafAbsC[2][1] + afEA[2]*aafAbsC[1][1];
    fR1 = afEB[0]*aafAbsC[0][2] + afEB[2]*aafAbsC[0][0];
    fR01 = fR0 + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*A0xB2
    fR = Math<Real>::FAbs(afAD[2]*aafC[1][2]-afAD[1]*aafC[2][2]);
    fR0 = afEA[1]*aafAbsC[2][2] + afEA[2]*aafAbsC[1][2];
    fR1 = afEB[0]*aafAbsC[0][1] + afEB[1]*aafAbsC[0][0];
    fR01 = fR0 + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*A1xB0
    fR = Math<Real>::FAbs(afAD[0]*aafC[2][0]-afAD[2]*aafC[0][0]);
    fR0 = afEA[0]*aafAbsC[2][0] + afEA[2]*aafAbsC[0][0];
    fR1 = afEB[1]*aafAbsC[1][2] + afEB[2]*aafAbsC[1][1];
    fR01 = fR0 + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*A1xB1
    fR = Math<Real>::FAbs(afAD[0]*aafC[2][1]-afAD[2]*aafC[0][1]);
    fR0 = afEA[0]*aafAbsC[2][1] + afEA[2]*aafAbsC[0][1];
    fR1 = afEB[0]*aafAbsC[1][2] + afEB[2]*aafAbsC[1][0];
    fR01 = fR0 + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*A1xB2
    fR = Math<Real>::FAbs(afAD[0]*aafC[2][2]-afAD[2]*aafC[0][2]);
    fR0 = afEA[0]*aafAbsC[2][2] + afEA[2]*aafAbsC[0][2];
    fR1 = afEB[0]*aafAbsC[1][1] + afEB[1]*aafAbsC[1][0];
    fR01 = fR0 + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*A2xB0
    fR = Math<Real>::FAbs(afAD[1]*aafC[0][0]-afAD[0]*aafC[1][0]);
    fR0 = afEA[0]*aafAbsC[1][0] + afEA[1]*aafAbsC[0][0];
    fR1 = afEB[1]*aafAbsC[2][2] + afEB[2]*aafAbsC[2][1];
    fR01 = fR0 + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*A2xB1
    fR = Math<Real>::FAbs(afAD[1]*aafC[0][1]-afAD[0]*aafC[1][1]);
    fR0 = afEA[0]*aafAbsC[1][1] + afEA[1]*aafAbsC[0][1];
    fR1 = afEB[0]*aafAbsC[2][2] + afEB[2]*aafAbsC[2][0];
    fR01 = fR0 + fR1;
    if (fR > fR01)
    {
        return false;
    }

    // axis C0+t*A2xB2
    fR = Math<Real>::FAbs(afAD[1]*aafC[0][2]-afAD[0]*aafC[1][2]);
    fR0 = afEA[0]*aafAbsC[1][2] + afEA[1]*aafAbsC[0][2];
    fR1 = afEB[0]*aafAbsC[2][1] + afEB[1]*aafAbsC[2][0];
    fR01 = fR0 + fR1;
    if (fR > fR01)
    {
        return false;
    }

    return true;
}
//----------------------------------------------------------------------------
template <class Real>
bool IntrBox3Box3<Real>::Test (Real fTMax,
    const Vector3<Real>& rkVelocity0, const Vector3<Real>& rkVelocity1)
{
    if (rkVelocity0 == rkVelocity1)
    {
        if (Test())
        {
            m_fContactTime = (Real)0.0;
            return true;
        }
        return false;
    }

    // Cutoff for cosine of angles between box axes.  This is used to catch
    // the cases when at least one pair of axes are parallel.  If this
    // happens, there is no need to include the cross-product axes for
    // separation.
    const Real fCutoff = (Real)1.0 - Math<Real>::ZERO_TOLERANCE;
    bool bExistsParallelPair = false;

    // convenience variables
    const Vector3<Real>* akA = m_pkBox0->Axis;
    const Vector3<Real>* akB = m_pkBox1->Axis;
    const Real* afEA = m_pkBox0->Extent;
    const Real* afEB = m_pkBox1->Extent;
    Vector3<Real> kD = m_pkBox1->Center - m_pkBox0->Center;
    Vector3<Real> kW = rkVelocity1 - rkVelocity0;
    Real aafC[3][3];     // matrix C = A^T B, c_{ij} = Dot(A_i,B_j)
    Real aafAbsC[3][3];  // |c_{ij}|
    Real afAD[3];        // Dot(A_i,D)
    Real afAW[3];        // Dot(A_i,W)
    Real fMin0, fMax0, fMin1, fMax1, fCenter, fRadius, fSpeed;
    int i, j;

    m_fContactTime = (Real)0.0;
    Real fTLast = Math<Real>::MAX_REAL;

    // axes C0+t*A[i]
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            aafC[i][j] = akA[i].Dot(akB[j]);
            aafAbsC[i][j] = Math<Real>::FAbs(aafC[i][j]);
            if (aafAbsC[i][j] > fCutoff)
            {
                bExistsParallelPair = true;
            }
        }
        afAD[i] = akA[i].Dot(kD);
        afAW[i] = akA[i].Dot(kW);
        fMin0 = -afEA[i];
        fMax0 = +afEA[i];
        fRadius = afEB[0]*aafAbsC[i][0] + afEB[1]*aafAbsC[i][1] +
            afEB[2]*aafAbsC[i][2];
        fMin1 = afAD[i] - fRadius;
        fMax1 = afAD[i] + fRadius;
        fSpeed = afAW[i];
        if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
        {
            return false;
        }
    }

    // axes C0+t*B[i]
    for (i = 0; i < 3; i++)
    {
        fRadius = afEA[0]*aafAbsC[0][i] + afEA[1]*aafAbsC[1][i] +
            afEA[2]*aafAbsC[2][i];
        fMin0 = -fRadius;
        fMax0 = +fRadius;
        fCenter = akB[i].Dot(kD);
        fMin1 = fCenter - afEB[i];
        fMax1 = fCenter + afEB[i];
        fSpeed = kW.Dot(akB[i]);
        if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
        {
            return false;
        }
    }

    // At least one pair of box axes was parallel, so the separation is
    // effectively in 2D where checking the "edge" normals is sufficient for
    // the separation of the boxes.
    if (bExistsParallelPair)
    {
        return true;
    }

    // axis C0+t*A0xB0
    fRadius = afEA[1]*aafAbsC[2][0] + afEA[2]*aafAbsC[1][0];
    fMin0 = -fRadius;
    fMax0 = +fRadius;
    fCenter = afAD[2]*aafC[1][0] - afAD[1]*aafC[2][0];
    fRadius = afEB[1]*aafAbsC[0][2] + afEB[2]*aafAbsC[0][1];
    fMin1 = fCenter - fRadius;
    fMax1 = fCenter + fRadius;
    fSpeed = afAW[2]*aafC[1][0] - afAW[1]*aafC[2][0];
    if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
    {
        return false;
    }

    // axis C0+t*A0xB1
    fRadius = afEA[1]*aafAbsC[2][1] + afEA[2]*aafAbsC[1][1];
    fMin0 = -fRadius;
    fMax0 = +fRadius;
    fCenter = afAD[2]*aafC[1][1] - afAD[1]*aafC[2][1];
    fRadius = afEB[0]*aafAbsC[0][2] + afEB[2]*aafAbsC[0][0];
    fMin1 = fCenter - fRadius;
    fMax1 = fCenter + fRadius;
    fSpeed = afAW[2]*aafC[1][1] - afAW[1]*aafC[2][1];
    if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
    {
        return false;
    }

    // axis C0+t*A0xB2
    fRadius = afEA[1]*aafAbsC[2][2] + afEA[2]*aafAbsC[1][2];
    fMin0 = -fRadius;
    fMax0 = +fRadius;
    fCenter = afAD[2]*aafC[1][2] - afAD[1]*aafC[2][2];
    fRadius = afEB[0]*aafAbsC[0][1] + afEB[1]*aafAbsC[0][0];
    fMin1 = fCenter - fRadius;
    fMax1 = fCenter + fRadius;
    fSpeed = afAW[2]*aafC[1][2] - afAW[1]*aafC[2][2];
    if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
    {
        return false;
    }

    // axis C0+t*A1xB0
    fRadius = afEA[0]*aafAbsC[2][0] + afEA[2]*aafAbsC[0][0];
    fMin0 = -fRadius;
    fMax0 = +fRadius;
    fCenter = afAD[0]*aafC[2][0] - afAD[2]*aafC[0][0];
    fRadius = afEB[1]*aafAbsC[1][2] + afEB[2]*aafAbsC[1][1];
    fMin1 = fCenter - fRadius;
    fMax1 = fCenter + fRadius;
    fSpeed = afAW[0]*aafC[2][0] - afAW[2]*aafC[0][0];
    if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
    {
        return false;
    }

    // axis C0+t*A1xB1
    fRadius = afEA[0]*aafAbsC[2][1] + afEA[2]*aafAbsC[0][1];
    fMin0 = -fRadius;
    fMax0 = +fRadius;
    fCenter = afAD[0]*aafC[2][1] - afAD[2]*aafC[0][1];
    fRadius = afEB[0]*aafAbsC[1][2] + afEB[2]*aafAbsC[1][0];
    fMin1 = fCenter - fRadius;
    fMax1 = fCenter + fRadius;
    fSpeed = afAW[0]*aafC[2][1] - afAW[2]*aafC[0][1];
    if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
    {
        return false;
    }

    // axis C0+t*A1xB2
    fRadius = afEA[0]*aafAbsC[2][2] + afEA[2]*aafAbsC[0][2];
    fMin0 = -fRadius;
    fMax0 = +fRadius;
    fCenter = afAD[0]*aafC[2][2] - afAD[2]*aafC[0][2];
    fRadius = afEB[0]*aafAbsC[1][1] + afEB[1]*aafAbsC[1][0];
    fMin1 = fCenter - fRadius;
    fMax1 = fCenter + fRadius;
    fSpeed = afAW[0]*aafC[2][2] - afAW[2]*aafC[0][2];
    if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
    {
        return false;
    }

    // axis C0+t*A2xB0
    fRadius = afEA[0]*aafAbsC[1][0] + afEA[1]*aafAbsC[0][0];
    fMin0 = -fRadius;
    fMax0 = +fRadius;
    fCenter = afAD[1]*aafC[0][0] - afAD[0]*aafC[1][0];
    fRadius = afEB[1]*aafAbsC[2][2] + afEB[2]*aafAbsC[2][1];
    fMin1 = fCenter - fRadius;
    fMax1 = fCenter + fRadius;
    fSpeed = afAW[1]*aafC[0][0] - afAW[0]*aafC[1][0];
    if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
    {
        return false;
    }

    // axis C0+t*A2xB1
    fRadius = afEA[0]*aafAbsC[1][1] + afEA[1]*aafAbsC[0][1];
    fMin0 = -fRadius;
    fMax0 = +fRadius;
    fCenter = afAD[1]*aafC[0][1] - afAD[0]*aafC[1][1];
    fRadius = afEB[0]*aafAbsC[2][2] + afEB[2]*aafAbsC[2][0];
    fMin1 = fCenter - fRadius;
    fMax1 = fCenter + fRadius;
    fSpeed = afAW[1]*aafC[0][1] - afAW[0]*aafC[1][1];
    if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
    {
        return false;
    }

    // axis C0+t*A2xB2
    fRadius = afEA[0]*aafAbsC[1][2] + afEA[1]*aafAbsC[0][2];
    fMin0 = -fRadius;
    fMax0 = +fRadius;
    fCenter = afAD[1]*aafC[0][2] - afAD[0]*aafC[1][2];
    fRadius = afEB[0]*aafAbsC[2][1] + afEB[1]*aafAbsC[2][0];
    fMin1 = fCenter - fRadius;
    fMax1 = fCenter + fRadius;
    fSpeed = afAW[1]*aafC[0][2] - afAW[0]*aafC[1][2];
    if (IsSeparated(fMin0,fMax0,fMin1,fMax1,fSpeed,fTMax,fTLast))
    {
        return false;
    }

    return true;
}
//----------------------------------------------------------------------------
template <class Real>
bool IntrBox3Box3<Real>::Find (Real fTMax, const Vector3<Real>& rkVelocity0,
    const Vector3<Real>& rkVelocity1)
{
    m_iQuantity = 0;

    m_fContactTime = (Real)0;
    Real fTLast = Math<Real>::MAX_REAL;

    // Relative velocity of box1 relative to box0.
    Vector3<Real> kVelocity = rkVelocity1 - rkVelocity0;

    int i0, i1;
    int eSide = IntrConfiguration<Real>::NONE;
    IntrConfiguration<Real> kBox0Cfg, kBox1Cfg;
    Vector3<Real> kAxis;

    // box 0 normals
    for (i0 = 0; i0 < 3; i0++)
    {
        kAxis = m_pkBox0->Axis[i0];
        if (!IntrAxis<Real>::Find(kAxis,*m_pkBox0,*m_pkBox1,kVelocity,fTMax,
            m_fContactTime,fTLast,eSide,kBox0Cfg,kBox1Cfg))
        {
            return false;
        }
    }

    // box 1 normals
    for (i1 = 0; i1 < 3; i1++)
    {
        kAxis = m_pkBox1->Axis[i1];
        if (!IntrAxis<Real>::Find(kAxis,*m_pkBox0,*m_pkBox1,kVelocity,fTMax,
            m_fContactTime,fTLast,eSide,kBox0Cfg,kBox1Cfg))
        {
            return false;
        }
    }

    // box 0 edges cross box 1 edges
    for (i0 = 0; i0 < 3; i0++)
    {
        for (i1 = 0; i1 < 3; i1++)
        {
            kAxis = m_pkBox0->Axis[i0].Cross(m_pkBox1->Axis[i1]);

            // Since all axes are unit length (assumed), then can just compare
            // against a constant (not relative) epsilon.
            if (kAxis.SquaredLength() <= Math<Real>::ZERO_TOLERANCE)
            {
                // Simple separation, axis i0 and i1 are parallel.  If any two
                // axes are parallel, then the only comparisons that need to
                // be done are between the faces themselves, which have
                // already been done.  Therefore, if they haven't been
                // separated yet, nothing else will.  Quick out.
                return true;
            }

            if (!IntrAxis<Real>::Find(kAxis,*m_pkBox0,*m_pkBox1,kVelocity,
                fTMax,m_fContactTime,fTLast,eSide,kBox0Cfg,kBox1Cfg))
            {
                return false;
            }
        }
    }

    // velocity cross box 0 edges
    for (i0 = 0; i0 < 3; i0++)
    {
        kAxis = kVelocity.Cross(m_pkBox0->Axis[i0]);
        if (!IntrAxis<Real>::Find(kAxis,*m_pkBox0,*m_pkBox1,kVelocity,fTMax,
            m_fContactTime,fTLast,eSide,kBox0Cfg,kBox1Cfg))
        {
            return false;
        }
    }

    // velocity cross box 1 edges
    for (i1 = 0; i1 < 3; i1++)
    {
        kAxis = kVelocity.Cross(m_pkBox1->Axis[i1]);
        if (!IntrAxis<Real>::Find(kAxis,*m_pkBox0,*m_pkBox1,kVelocity,fTMax,
            m_fContactTime,fTLast,eSide,kBox0Cfg,kBox1Cfg))
        {
            return false;
        }
    }

    if (m_fContactTime <= (Real)0 || eSide == IntrConfiguration<Real>::NONE)
    {
        return false;
    }

    FindContactSet<Real>(*m_pkBox0,*m_pkBox1,eSide,kBox0Cfg,kBox1Cfg,
        rkVelocity0,rkVelocity1,m_fContactTime,m_iQuantity,m_akPoint);

    return true;
}
//----------------------------------------------------------------------------
template <class Real>
bool IntrBox3Box3<Real>::Test (Real fTMax, int iNumSteps,
    const Vector3<Real>& rkVelocity0, const Vector3<Real>& rkRotCenter0,
    const Vector3<Real>& rkRotAxis0, const Vector3<Real>& rkVelocity1,
    const Vector3<Real>& rkRotCenter1, const Vector3<Real>& rkRotAxis1)
{
    // The time step for the integration.
    Real fStep = fTMax/(Real)iNumSteps;

    // Initialize subinterval boxes.
    Box3<Real> kSubBox0, kSubBox1;
    kSubBox0.Center = m_pkBox0->Center;
    kSubBox1.Center = m_pkBox1->Center;
    int i;
    for (i = 0; i < 3; i++)
    {
        kSubBox0.Axis[i] = m_pkBox0->Axis[i];
        kSubBox1.Axis[i] = m_pkBox1->Axis[i];
    }

    // Integrate the differential equations using Euler's method.
    for (int iStep = 1; iStep <= iNumSteps; iStep++)
    {
        // Compute box velocities and test boxes for intersection.
        Real fSubTime = fStep*(Real)iStep;
        Vector3<Real> kNewRotCenter0 = rkRotCenter0 + fSubTime*rkVelocity0;
        Vector3<Real> kNewRotCenter1 = rkRotCenter1 + fSubTime*rkVelocity1;
        Vector3<Real> kDiff0 = kSubBox0.Center - kNewRotCenter0;
        Vector3<Real> kDiff1 = kSubBox1.Center - kNewRotCenter1;
        Vector3<Real> kSubVelocity0 =
            fStep*(rkVelocity0 + rkRotAxis0.Cross(kDiff0));
        Vector3<Real> kSubVelocity1 =
            fStep*(rkVelocity1 + rkRotAxis1.Cross(kDiff1));

        IntrBox3Box3 kCalc(kSubBox0,kSubBox1);
        if (kCalc.Test(fStep,kSubVelocity0,kSubVelocity1))
        {
            return true;
        }

        // Update the box centers.
        kSubBox0.Center = kSubBox0.Center + kSubVelocity0;
        kSubBox1.Center = kSubBox1.Center + kSubVelocity1;

        // Update the box axes.
        for (i = 0; i < 3; i++)
        {
            kSubBox0.Axis[i] = kSubBox0.Axis[i] +
                fStep*rkRotAxis0.Cross(kSubBox0.Axis[i]);

            kSubBox1.Axis[i] = kSubBox1.Axis[i] +
                fStep*rkRotAxis1.Cross(kSubBox1.Axis[i]);
        }

        // Use Gram-Schmidt to orthonormalize the updated axes.  NOTE:  If
        // T/N is small and N is small, you can remove this expensive step
        // with the assumption that the updated axes are nearly orthonormal.
        Vector3<Real>::Orthonormalize(kSubBox0.Axis);
        Vector3<Real>::Orthonormalize(kSubBox1.Axis);
    }

    // NOTE:  If the boxes do not intersect, then the application might want
    // to move/rotate the boxes to their new locations.  In this case you
    // want to return the final values of kSubBox0 and kSubBox1 so that the
    // application can set rkBox0 <- kSubBox0 and rkBox1 <- kSubBox1.
    // Otherwise, the application would have to solve the differential
    // equation again or compute the new box locations using the closed form
    // solution for the rigid motion.

    return false;
}
//----------------------------------------------------------------------------
template <class Real>
bool IntrBox3Box3<Real>::IsSeparated (Real fMin0, Real fMax0, Real fMin1,
    Real fMax1, Real fSpeed, Real fTMax, Real& rfTLast)
{
    Real fInvSpeed, fT;

    if (fMax1 < fMin0) // box1 initially on left of box0
    {
        if (fSpeed <= (Real)0)
        {
            // The projection intervals are moving apart.
            return true;
        }
        fInvSpeed = ((Real)1)/fSpeed;

        fT = (fMin0 - fMax1)*fInvSpeed;
        if (fT > m_fContactTime)
        {
            m_fContactTime = fT;
        }

        if (m_fContactTime > fTMax)
        {
            // Intervals do not intersect during the specified time.
            return true;
        }

        fT = (fMax0 - fMin1)*fInvSpeed;
        if (fT < rfTLast)
        {
            rfTLast = fT;
        }

        if (m_fContactTime > rfTLast)
        {
            // Physically inconsistent times--the objects cannot intersect.
            return true;
        }
    }
    else if (fMax0 < fMin1) // box1 initially on right of box0
    {
        if (fSpeed >= (Real)0)
        {
            // The projection intervals are moving apart.
            return true;
        }
        fInvSpeed = ((Real)1)/fSpeed;

        fT = (fMax0 - fMin1)*fInvSpeed;
        if (fT > m_fContactTime)
        {
            m_fContactTime = fT;
        }

        if (m_fContactTime > fTMax)
        {
            // Intervals do not intersect during the specified time.
            return true;
        }

        fT = (fMin0 - fMax1)*fInvSpeed;
        if (fT < rfTLast)
        {
            rfTLast = fT;
        }

        if (m_fContactTime > rfTLast)
        {
            // Physically inconsistent times--the objects cannot intersect.
            return true;
        }
    }
    else // box0 and box1 initially overlap
    {
        if (fSpeed > (Real)0)
        {
            fT = (fMax0 - fMin1)/fSpeed;
            if (fT < rfTLast)
            {
                rfTLast = fT;
            }

            if (m_fContactTime > rfTLast)
            {
                // Physically inconsistent times--the objects cannot
                // intersect.
                return true;
            }
        }
        else if (fSpeed < (Real)0)
        {
            fT = (fMin0 - fMax1)/fSpeed;
            if (fT < rfTLast)
            {
                rfTLast = fT;
            }

            if (m_fContactTime > rfTLast)
            {
                // Physically inconsistent times--the objects cannot
                // intersect.
                return true;
            }
        }
    }

    return false;
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// explicit instantiation
//----------------------------------------------------------------------------
template WM4_FOUNDATION_ITEM
class IntrBox3Box3<float>;

template WM4_FOUNDATION_ITEM
class IntrBox3Box3<double>;
//----------------------------------------------------------------------------
}
