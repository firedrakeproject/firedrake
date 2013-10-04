/*  Copyright (C) 2006 Imperial College London and others.

    Please see the AUTHORS file in the main source directory for a full list
    of copyright holders.

    Prof. C Pain
    Applied Modelling and Computation Group
    Department of Earth Science and Engineering
    Imperial College London

    C.Pain@Imperial.ac.uk

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation,
    version 2.1 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
    USA
*/

#ifndef SPUD_ENUMS_H
#define SPUD_ENUMS_H

#ifdef __cplusplus
namespace Spud{
#endif

#ifdef __cplusplus
  enum OptionType{
#else
  enum SpudOptionType{
#endif
    SPUD_DOUBLE = 0,
    SPUD_INT    = 1,
    SPUD_NONE   = 2,
    SPUD_STRING = 3,
  };

#ifdef __cplusplus
  enum OptionError{
#else
  enum SpudOptionError{
#endif
    SPUD_NO_ERROR                = 0,
    SPUD_KEY_ERROR               = 1,
    SPUD_TYPE_ERROR              = 2,
    SPUD_RANK_ERROR              = 3,
    SPUD_SHAPE_ERROR             = 4,
    SPUD_FILE_ERROR              = 5,
    SPUD_NEW_KEY_WARNING         = -1,
    SPUD_ATTR_SET_FAILED_WARNING = -2,
  };

#ifdef __cplusplus
}
#endif

#endif
