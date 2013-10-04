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
// Version: 4.7.0 (2008/09/15)

#ifndef WM4INTRUTILITY3_H
#define WM4INTRUTILITY3_H

#include "Wm4FoundationLIB.h"
#include "Wm4Box3.h"
#include "Wm4Segment3.h"
#include "Wm4Triangle3.h"

namespace Wm4
{

//----------------------------------------------------------------------------
template <class Real>
class WM4_FOUNDATION_ITEM IntrConfiguration
{
public:
    // ContactSide (order of the intervals of projection).
    enum
    {
        LEFT,
        RIGHT,
        NONE
    };

    // VertexProjectionMap (how the vertices are projected to the minimum
    // and maximum points of the interval).
    enum
    {
        m2, m11,             // segments
        m3, m21, m12, m111,  // triangles
        m44, m2_2, m1_1      // boxes
    };

    // The VertexProjectionMap value for the configuration.
    int Map;

    // The order of the vertices.
    int Index[8];

    // Projection interval.
    Real Min, Max;
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
template <class Real>
class WM4_FOUNDATION_ITEM IntrAxis
{
public:
    // Test-query for intersection of projected intervals.  The velocity
    // input is the difference objectVelocity1 - objectVelocity0.  The
    // first and last times of contact are computed.
    static bool Test (const Vector3<Real>& rkAxis,
        const Vector3<Real> akSegment[2], const Triangle3<Real>& rkTriangle,
        const Vector3<Real>& rkVelocity, Real fTMax, Real& rfTFirst,
        Real& rfTLast);

    static bool Test (const Vector3<Real>& rkAxis,
        const Vector3<Real> akSegment[2], const Box3<Real>& rkBox,
        const Vector3<Real>& rkVelocity, Real fTMax, Real& rfTFirst,
        Real& rfTLast);

    static bool Test (const Vector3<Real>& rkAxis,
        const Triangle3<Real>& rkTriangle, const Box3<Real>& rkBox,
        const Vector3<Real>& rkVelocity, Real fTMax, Real& rfTFirst,
        Real& rfTLast);

    static bool Test (const Vector3<Real>& rkAxis,
        const Box3<Real>& rkBox0, const Box3<Real>& rkBox1,
        const Vector3<Real>& rkVelocity, Real fTMax, Real& rfTFirst,
        Real& rfTLast);

    // Find-query for intersection of projected intervals.  The velocity
    // input is the difference objectVelocity1 - objectVelocity0.  The
    // first and last times of contact are computed, as is information about
    // the contact configuration and the ordering of the projections (the
    // contact side).
    static bool Find (const Vector3<Real>& rkAxis,
        const Vector3<Real> akSegment[2], const Triangle3<Real>& rkTriangle,
        const Vector3<Real>& rkVelocity, Real fTMax, Real& rfTFirst,
        Real& rfTLast, int& reSide, IntrConfiguration<Real>& rkSegCfgFinal,
        IntrConfiguration<Real>& rkTriCfgFinal);

    static bool Find (const Vector3<Real>& rkAxis,
        const Vector3<Real> akSegment[2], const Box3<Real>& rkBox,
        const Vector3<Real>& rkVelocity, Real fTMax, Real& rfTFirst,
        Real& rfTLast, int& reSide, IntrConfiguration<Real>& rkSegCfgFinal,
        IntrConfiguration<Real>& rkBoxCfgFinal);

    static bool Find (const Vector3<Real>& rkAxis,
        const Triangle3<Real>& rkTriangle, const Box3<Real>& rkBox,
        const Vector3<Real>& rkVelocity, Real fTMax, Real& rfTFirst,
        Real& rfTLast, int& reSide, IntrConfiguration<Real>& rkTriCfgFinal,
        IntrConfiguration<Real>& rkBoxCfgFinal);

    static bool Find (const Vector3<Real>& rkAxis,
        const Box3<Real>& rkBox0, const Box3<Real>& rkBox1,
        const Vector3<Real>& rkVelocity, Real fTMax, Real& rfTFirst,
        Real& rfTLast, int& reSide, IntrConfiguration<Real>& rkBox0CfgFinal,
        IntrConfiguration<Real>& rkBox1CfgFinal);

    // Projections.
    static void GetProjection (const Vector3<Real>& rkAxis,
        const Vector3<Real> akSegment[2], Real& rfMin, Real& rfMax);

    static void GetProjection (const Vector3<Real>& rkAxis,
        const Triangle3<Real>& rkTriangle, Real& rfMin, Real& rfMax);

    static void GetProjection (const Vector3<Real>& rkAxis,
        const Box3<Real>& rkBox, Real& rfMin, Real& rfMax);

    // Configurations.
    static void GetConfiguration (const Vector3<Real>& rkAxis,
        const Vector3<Real> akSegment[2], IntrConfiguration<Real>& rkCfg);

    static void GetConfiguration (const Vector3<Real>& rkAxis,
        const Triangle3<Real>& rkTriangle, IntrConfiguration<Real>& rkCfg);

    static void GetConfiguration (const Vector3<Real>& rkAxis,
        const Box3<Real>& rkBox, IntrConfiguration<Real>& rkCfg);

    // Low-level test-query for projections.
    static bool Test (const Vector3<Real>& rkAxis,
        const Vector3<Real>& rkVelocity, Real fMin0, Real fMax0, Real fMin1,
        Real fMax1, Real fTMax, Real& rfTFirst, Real& rfTLast);

    // Low-level find-query for projections.
    static bool Find (const Vector3<Real>& rkAxis,
        const Vector3<Real>& rkVelocity,
        const IntrConfiguration<Real>& rkCfg0Start,
        const IntrConfiguration<Real>& rkCfg1Start, Real fTMax,
        int& rkSide, IntrConfiguration<Real>& rkCfg0Final,
        IntrConfiguration<Real>& rkCfg1Final, Real& rfTFirst, Real& rfTLast);
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
template <class Real>
class WM4_FOUNDATION_ITEM FindContactSet
{
public:
    FindContactSet (const Vector3<Real> akSegment[2],
        const Triangle3<Real>& rkTriangle, int eSide,
        const IntrConfiguration<Real>& rkSegCfg,
        const IntrConfiguration<Real>& rkTriCfg,
        const Vector3<Real>& rkSegVelocity,
        const Vector3<Real>& rkTriVelocity, Real fTFirst, int& riQuantity,
        Vector3<Real>* akP);

    FindContactSet (const Vector3<Real> akSegment[2], const Box3<Real>& rkBox,
        int eSide, const IntrConfiguration<Real>& rkSegCfg,
        const IntrConfiguration<Real>& rkBoxCfg,
        const Vector3<Real>& rkSegVelocity,
        const Vector3<Real>& rkBoxVelocity, Real fTFirst, int& riQuantity,
        Vector3<Real>* akP);

    FindContactSet (const Triangle3<Real>& rkTriangle,
        const Box3<Real>& rkBox, int eSide,
        const IntrConfiguration<Real>& rkTriCfg,
        const IntrConfiguration<Real>& rkBoxCfg,
        const Vector3<Real>& rkTriVelocity,
        const Vector3<Real>& rkBoxVelocity, Real fTFirst, int& riQuantity,
        Vector3<Real>* akP);

    FindContactSet (const Box3<Real>& rkBox0, const Box3<Real>& rkBox1,
        int eSide, const IntrConfiguration<Real>& rkBox0Cfg,
        const IntrConfiguration<Real>& rkBox1Cfg,
        const Vector3<Real>& rkBox0Velocity,
        const Vector3<Real>& rkBox1Velocity, Real fTFirst, int& riQuantity,
        Vector3<Real>* akP);

private:
    // These functions are called when it is known that the features are
    // intersecting.  Consequently, they are specialized versions of the
    // object-object intersection algorithms.

    static void ColinearSegments (const Vector3<Real> akSegment0[2],
        const Vector3<Real> akSegment1[2], int& riQuantity,
        Vector3<Real>* akP);

    static void SegmentThroughPlane (const Vector3<Real> akSegment[2],
        const Vector3<Real>& rkPlaneOrigin, const Vector3<Real>& rkPlaneNormal,
        int& riQuantity, Vector3<Real>* akP);

    static void SegmentSegment (const Vector3<Real> akSegment0[2],
        const Vector3<Real> akSegment1[2], int& riQuantity,
        Vector3<Real>* akP);

    static void ColinearSegmentTriangle (const Vector3<Real> akSegment[2],
        const Vector3<Real> akTriangle[3], int& riQuantity,
        Vector3<Real>* akP);

    static void CoplanarSegmentRectangle (const Vector3<Real> akSegment[2],
        const Vector3<Real> akRectangle[4], int& riQuantity,
        Vector3<Real>* akP);

    static void CoplanarTriangleRectangle (const Vector3<Real> akTriangle[3],
        const Vector3<Real> akRectangle[4], int& riQuantity,
        Vector3<Real>* akP);

    static void CoplanarRectangleRectangle (
        const Vector3<Real> akRectangle0[4],
        const Vector3<Real> akRectangle1[4], int& riQuantity,
        Vector3<Real>* akP);
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Miscellaneous support.
//----------------------------------------------------------------------------
// The input and output polygons are stored in akP.  The size of akP is
// assumed to be large enough to store the clipped convex polygon vertices.
// For now the maximum array size is 8 to support the current intersection
// algorithms.
template <class Real> WM4_FOUNDATION_ITEM
void ClipConvexPolygonAgainstPlane (const Vector3<Real>& rkNormal,
    Real fConstant, int& riQuantity, Vector3<Real>* akP);

// Translates an index into the box back into real coordinates.
template <class Real> WM4_FOUNDATION_ITEM
Vector3<Real> GetPointFromIndex (int iIndex, const Box3<Real>& rkBox);
//----------------------------------------------------------------------------
}

#endif
