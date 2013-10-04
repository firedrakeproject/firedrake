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

#ifndef WM4INTRBOX3BOX3_H
#define WM4INTRBOX3BOX3_H

#include "Wm4FoundationLIB.h"
#include "Wm4Intersector.h"
#include "Wm4Box3.h"

namespace Wm4
{

template <class Real>
class WM4_FOUNDATION_ITEM IntrBox3Box3
    : public Intersector<Real,Vector3<Real> >
{
public:
    IntrBox3Box3 (const Box3<Real>& rkBox0, const Box3<Real>& rkBox1);

    // Object access.
    const Box3<Real>& GetBox0 () const;
    const Box3<Real>& GetBox1 () const;

    // Static test-intersection query.
    virtual bool Test ();

    // Dynamic test-intersection query.  The first time of contact (if any)
    // is computed, but not any information about the contact set.
    virtual bool Test (Real fTMax, const Vector3<Real>& rkVelocity0,
        const Vector3<Real>& rkVelocity1);

    // Dynamic find-intersection query.  The contact set is computed.
    virtual bool Find (Real fTMax, const Vector3<Real>& rkVelocity0,
        const Vector3<Real>& rkVelocity1);

    // Dynamic test-intersection query where the boxes have constant linear
    // velocities *and* constant angular velocities.  The length of the
    // rotation axes are the angular speeds.  A differential equation solver
    // is used to predict the intersection.  The input iNumSteps is the
    // number of iterations for the numerical ODE solver.
    bool Test (Real fTMax, int iNumSteps, const Vector3<Real>& rkVelocity0,
        const Vector3<Real>& rkRotCenter0, const Vector3<Real>& rkRotAxis0,
        const Vector3<Real>& rkVelocity1, const Vector3<Real>& rkRotCenter1,
        const Vector3<Real>& rkRotAxis1);

private:
    using Intersector<Real,Vector3<Real> >::m_fContactTime;

    // Support for dynamic queries.  The inputs are the projection intervals
    // for the boxes onto a potential separating axis, the relative speed of
    // the intervals, and the maximum time for the query.  The outputs are
    // the first time when separating fails and the last time when separation
    // begins again along that axis.  The outputs are *updates* in the sense
    // that this function is called repeatedly for the potential separating
    // axes.  The output first time is updated only if it is larger than
    // the input first time.  The output last time is updated only if it is
    // smaller than the input last time.
    //
    // NOTE:  The BoxBoxAxisTest function could be used, but the box-box
    // code optimizes the projections of the boxes onto the various axes.
    // This function is effectively BoxBoxAxisTest but without the dot product
    // of axis-direction and velocity to obtain speed.  The optimizations are
    // to compute the speed with fewer computations.
    bool IsSeparated (Real fMin0, Real fMax0, Real fMin1, Real fMax1,
        Real fSpeed, Real fTMax, Real& rfTLast);

    // The objects to intersect.
    const Box3<Real>* m_pkBox0;
    const Box3<Real>* m_pkBox1;

    // The intersection set for dynamic find-intersection.  The worst case
    // is a polygon with 8 vertices.
    int m_iQuantity;
    Vector3<Real> m_akPoint[8];
};

typedef IntrBox3Box3<float> IntrBox3Box3f;
typedef IntrBox3Box3<double> IntrBox3Box3d;

}

#endif
