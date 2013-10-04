!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Prof. C Pain
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineering
!    Imperial College London
!
!    amcgsoftware@imperial.ac.uk
!
!    This library is free software; you can redistribute it and/or
!    modify it under the terms of the GNU Lesser General Public
!    License as published by the Free Software Foundation,
!    version 2.1 of the License.
!
!    This library is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!    Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public
!    License along with this library; if not, write to the Free Software
!    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
!    USA

#ifndef DEBUG_H
#define DEBUG_H
#include "confdefs.h"

#ifndef __FILE__
#error __FILE__ does not work
#endif

#ifndef __LINE__
#error __LINE__ does not work
#endif

#define ewrite(priority, format) if (priority <= current_debug_level) write(debug_unit(priority), format)
#define EWRITE(priority, format) ewrite(priority, format)

#ifdef __GNUC__
! gfortran/gccs traditional cpp does not allow #array stringification
!#define ewrite_minmax(array) ewrite(2, *) "Min, max of "//'array'//" = ",minval(array), maxval(array)
#define ewrite_minmax(array) if (current_debug_level >= 2) call write_minmax(array, 'array')
#else
!#define ewrite_minmax(array) ewrite(2, *) "Min, max of "//#array//" = ",minval(array), maxval(array)
#define ewrite_minmax(array) if (current_debug_level >= 2) call write_minmax(array, #array)
#endif

#define EWRITE_MINMAX(array) ewrite_minmax(array)

#define ploc(x, i) call dprintf(-1, "%d: %p\n$", (i), (x))

#define FLAbort(X) call FLAbort_pinpoint(X, __FILE__, __LINE__)
#define FLExit(X) call FLExit_pinpoint(X, __FILE__, __LINE__)

! #define FORTRAN_DISALLOWS_LONG_LINES
#ifdef NDEBUG
#define ASSERT(X)
#else
#ifdef FORTRAN_DISALLOWS_LONG_LINES
#define ASSERT(X) IF(.NOT.(X)) FLAbort('Failed assertion ')
#else
#define ASSERT(X) IF(.NOT.(X)) FLAbort('Failed assertion '//'X')
#endif
#endif
#define assert(X) ASSERT(X)

#ifdef DDEBUG
#define METRIC_DEBUG
#endif

#endif
