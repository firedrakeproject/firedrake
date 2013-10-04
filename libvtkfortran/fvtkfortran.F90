!!$ Copyright (C) 2004- Imperial College London and others.
!!$
!!$   Please see the AUTHORS file in the main source directory for a full
!!$   list of copyright holders.
!!$
!!$   Adrian Umpleby
!!$   Applied Modelling and Computation Group
!!$   Department of Earth Science and Engineering
!!$   Imperial College London
!!$
!!$   adrian@imperial.ac.uk
!!$
!!$   This library is free software; you can redistribute it and/or
!!$   modify it under the terms of the GNU Lesser General Public
!!$   License as published by the Free Software Foundation; either
!!$   version 2.1 of the License.
!!$
!!$   This library is distributed in the hope that it will be useful,
!!$   but WITHOUT ANY WARRANTY; without even the implied warranty of
!!$   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!!$   Lesser General Public License for more details.
!!$
!!$   You should have received a copy of the GNU Lesser General Public
!!$   License along with this library; if not, write to the Free Software
!!$   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
!!$   USA

module vtkfortran
  !!< This module merely contains explicit interfaces to allow the
  !!< convenient use of vtkfortran in fortran.
  use iso_c_binding

  private

  ! Element types from VTK
  integer, public, parameter :: VTK_VERTEX=1
  integer, public, parameter :: VTK_POLY_VERTEX=2
  integer, public, parameter :: VTK_LINE=3
  integer, public, parameter :: VTK_POLY_LINE=4
  integer, public, parameter :: VTK_TRIANGLE=5
  integer, public, parameter :: VTK_TRIANGLE_STRIP=6
  integer, public, parameter :: VTK_POLYGON=7
  integer, public, parameter :: VTK_PIXEL=8
  integer, public, parameter :: VTK_QUAD=9
  integer, public, parameter :: VTK_TETRA=10
  integer, public, parameter :: VTK_VOXEL=11
  integer, public, parameter :: VTK_HEXAHEDRON=12
  integer, public, parameter :: VTK_WEDGE=13
  integer, public, parameter :: VTK_PYRAMID=14

  integer, public, parameter :: VTK_QUADRATIC_EDGE=21
  integer, public, parameter :: VTK_QUADRATIC_TRIANGLE=22
  integer, public, parameter :: VTK_QUADRATIC_QUAD=23
  integer, public, parameter :: VTK_QUADRATIC_TETRA=24
  integer, public, parameter :: VTK_QUADRATIC_HEXAHEDRON=25

  public :: vtkopen, vtkclose, vtkpclose, vtkwritemesh, vtkwritesn,&
       & vtkwritesc, vtkwritevn, vtkwritevc, vtkwritetn, vtkwritetc, &
       & vtksetactivescalars, vtksetactivevectors, &
       & vtksetactivetensors

  interface vtkopen
     subroutine vtkopen_c(outName, len1, vtkTitle, len2) bind(c,name="vtkopen")
       use iso_c_binding
       implicit none
       character(kind=c_char,len=1), dimension(*) :: outName
       integer(kind=c_int) :: len1
       character(kind=c_char,len=1), dimension(*) :: vtkTitle
       integer(kind=c_int) :: len2
     end subroutine vtkopen_c
     module procedure vtkopen_f90
  end interface

  interface vtkclose
     ! Close the current vtk file.
     subroutine vtkclose() bind(c)
     end subroutine vtkclose
  end interface

  interface vtkpclose
     ! Close the current vtk file - creates a parallel file.
     subroutine vtkpclose(rank, npartitions) bind(c)
       use iso_c_binding
       implicit none
       integer(kind=c_int) :: rank, npartitions
     end subroutine vtkpclose
  end interface

  interface vtkwritemesh
     ! Write mesh information to the current vtk file.
     SUBROUTINE VTKWRITEMESH(NNodes, NElems, x, y, z, enlist, &
          elementTypes, elementSizes) bind(c)
       use iso_c_binding
       implicit none
       integer(kind=c_int) :: NNodes
       integer(kind=c_int) :: NElems
       REAL(c_float) :: x(*)
       REAL(c_float) :: y(*)
       REAL(c_float) :: z(*)
       integer(kind=c_int) :: enlist(*)
       integer(kind=c_int) :: elementTypes(*)
       integer(kind=c_int) :: elementSizes(*)
     end SUBROUTINE VTKWRITEMESH
     SUBROUTINE VTKWRITEMESHD(NNodes, NElems, x, y, z, enlist, &
          elementTypes, elementSizes)  bind(c)
       use iso_c_binding
       implicit none
       integer(kind=c_int) :: NNodes
       integer(kind=c_int) :: NElems
       REAL(c_double) :: x(*)
       REAL(c_double) :: y(*)
       REAL(c_double) :: z(*)
       integer(kind=c_int) :: enlist(*)
       integer(kind=c_int) :: elementTypes(*)
       integer(kind=c_int) :: elementSizes(*)
     end SUBROUTINE VTKWRITEMESHD
  end interface

  interface vtkwritesn
     ! Write a scalar field to the current vtk file.
     SUBROUTINE VTKWRITEISN_C(vect, name, len) bind(c,name="vtkwriteisn")
       use iso_c_binding
       implicit none
       integer(kind=c_int) :: vect(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEISN_C
     SUBROUTINE VTKWRITEFSN_C(vect, name, len) bind(c,name="vtkwritefsn")
       use iso_c_binding
       implicit none
       REAL(c_float) :: vect(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEFSN_C
     SUBROUTINE VTKWRITEDSN_C(vect, name, len) bind(c,name="vtkwritedsn")
       use iso_c_binding
       implicit none
       REAL(c_double) :: vect(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEDSN_C
     module procedure vtkwriteisn_f90, vtkwritefsn_f90, vtkwritedsn_f90
  end interface

  interface vtkwritesc
     ! Write a scalar field (cell-based) to the current vtk file.
     SUBROUTINE VTKWRITEISC_C(vect, name, len) bind(c,name="vtkwriteisc")
       use iso_c_binding
       implicit none
       integer(kind=c_int) :: vect(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEISC_C
     SUBROUTINE VTKWRITEFSC_C(vect, name, len) bind(c,name="vtkwritefsc")
       use iso_c_binding
       implicit none
       REAL(c_float) :: vect(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEFSC_C
     SUBROUTINE VTKWRITEDSC_C(vect, name, len) bind(c,name="vtkwritedsc")
       use iso_c_binding
       implicit none
       REAL(c_double) :: vect(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEDSC_C
     module procedure vtkwriteisc_f90, vtkwritefsc_f90, vtkwritedsc_f90
  end interface

  interface vtkwritevn
     ! Write a vector field to the current vtk file.
     SUBROUTINE VTKWRITEFVN_C(vx, vy, vz, name, len) bind(c,name="vtkwritefvn")
       use iso_c_binding
       implicit none
       REAL(c_float) :: vx(*)
       REAL(c_float) :: vy(*)
       REAL(c_float) :: vz(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEFVN_C
     SUBROUTINE VTKWRITEDVN_C(vx, vy, vz, name, len) bind(c,name="vtkwritedvn")
       use iso_c_binding
       implicit none
       REAL(c_double) :: vx(*)
       REAL(c_double) :: vy(*)
       REAL(c_double) :: vz(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEDVN_C
     module procedure vtkwritefvn_f90, vtkwritedvn_f90
  end interface

  interface vtkwritevc
     ! Write a vector field (cell-based) to the current vtk file.
     SUBROUTINE VTKWRITEFVC_C(vx, vy, vz, name, len) bind(c,name="vtkwritefvc")
       use iso_c_binding
       implicit none
       REAL(c_float) :: vx(*)
       REAL(c_float) :: vy(*)
       REAL(c_float) :: vz(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEFVC_C
     SUBROUTINE VTKWRITEDVC_C(vx, vy, vz, name, len) bind(c,name="vtkwritedvc")
       use iso_c_binding
       implicit none
       REAL(c_double) :: vx(*)
       REAL(c_double) :: vy(*)
       REAL(c_double) :: vz(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEDVC_C
     module procedure vtkwritefvc_f90, vtkwritedvc_f90
  end interface

  interface vtkwritetn
     ! Write a tensor field to the current vtk file.
     SUBROUTINE VTKWRITEFTN_C(v1, v2, v3, v4, v5, v6, v7, v8, v9, name,&
          & len) bind(c,name="vtkwriteftn")
       use iso_c_binding
       implicit none
       REAL(c_float) :: v1(*)
       REAL(c_float) :: v2(*)
       REAL(c_float) :: v3(*)
       REAL(c_float) :: v4(*)
       REAL(c_float) :: v5(*)
       REAL(c_float) :: v6(*)
       REAL(c_float) :: v7(*)
       REAL(c_float) :: v8(*)
       REAL(c_float) :: v9(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEFTN_C
     SUBROUTINE VTKWRITEDTN_C(v1, v2, v3, v4, v5, v6, v7, v8, v9, name, len)&
          & bind(c,name="vtkwritedtn")
       use iso_c_binding
       implicit none
       REAL(c_double) :: v1(*)
       REAL(c_double) :: v2(*)
       REAL(c_double) :: v3(*)
       REAL(c_double) :: v4(*)
       REAL(c_double) :: v5(*)
       REAL(c_double) :: v6(*)
       REAL(c_double) :: v7(*)
       REAL(c_double) :: v8(*)
       REAL(c_double) :: v9(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEDTN_C
     module procedure vtkwriteftn_f90, vtkwritedtn_f90
  end interface

  interface vtkwritetc
     ! Write a tensor field (cell-based) to the current vtk file.
     SUBROUTINE VTKWRITEFTC_C(v1, v2, v3, v4, v5, v6, v7, v8, v9, name,&
          & len) bind(c,name="vtkwriteftc")
       use iso_c_binding
       implicit none
       REAL(c_float) :: v1(*)
       REAL(c_float) :: v2(*)
       REAL(c_float) :: v3(*)
       REAL(c_float) :: v4(*)
       REAL(c_float) :: v5(*)
       REAL(c_float) :: v6(*)
       REAL(c_float) :: v7(*)
       REAL(c_float) :: v8(*)
       REAL(c_float) :: v9(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEFTC_C
     SUBROUTINE VTKWRITEDTC_C(v1, v2, v3, v4, v5, v6, v7, v8, v9, name,&
          & len) bind(c,name="vtkwritedtc")
       use iso_c_binding
       implicit none
       REAL(c_double) :: v1(*)
       REAL(c_double) :: v2(*)
       REAL(c_double) :: v3(*)
       REAL(c_double) :: v4(*)
       REAL(c_double) :: v5(*)
       REAL(c_double) :: v6(*)
       REAL(c_double) :: v7(*)
       REAL(c_double) :: v8(*)
       REAL(c_double) :: v9(*)
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end SUBROUTINE VTKWRITEDTC_C
     module procedure vtkwriteftc_f90, vtkwritedtc_f90
  end interface

  interface vtksetactivescalars
    subroutine vtksetactivescalars_c(name, len) bind(c,name="vtksetactivescalars")
       use iso_c_binding
       implicit none
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end subroutine vtksetactivescalars_c
     module procedure vtksetactivescalars_f90
  end interface

  interface vtksetactivevectors
     subroutine vtksetactivevectors_c(name, len) bind(c,name="vtksetactivevectors")
       use iso_c_binding
       implicit none
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end subroutine vtksetactivevectors_c
     module procedure vtksetactivevectors_f90
  end interface

  interface vtksetactivetensors
     subroutine vtksetactivetensors_c(name, len) bind(c,name="vtksetactivetensors")
       use iso_c_binding
       implicit none
       character(kind=c_char,len=1), dimension(*) :: name
       integer(kind=c_int) :: len
     end subroutine vtksetactivetensors_c
     module procedure vtksetactivetensors_f90
  end interface


contains

  subroutine vtkopen_f90(outName, vtkTitle)
    ! Wrapper routine with nicer interface.
    character(len=*), intent(in) :: outName, vtkTitle

    call vtkopen_c(outName, len(outName), vtkTitle, len(vtkTitle))

  end subroutine vtkopen_f90

  subroutine vtkwriteisn_f90(vect, name)
    ! Wrapper routine with nicer interface.
    integer, intent(in) :: vect(*)
    character(len=*), intent(in) :: name

    call vtkwriteisn_c(vect, name, len(name))

  end subroutine vtkwriteisn_f90

  subroutine vtkwritefsn_f90(vect, name)
    ! Wrapper routine with nicer interface.
    real(c_float), intent(in) :: vect(*)
    character(len=*), intent(in) :: name

    call vtkwritefsn_c(vect, name, len(name))

  end subroutine vtkwritefsn_f90

  subroutine vtkwritedsn_f90(vect, name)
    ! Wrapper routine with nicer interface.
    real(c_double), intent(in) :: vect(*)
    character(len=*), intent(in) :: name

    call vtkwritedsn_c(vect, name, len(name))
  end subroutine vtkwritedsn_f90

  subroutine vtkwriteisc_f90(vect, name)
    ! Wrapper routine with nicer interface.
    integer, intent(in) :: vect(*)
    character(len=*), intent(in) :: name

    call vtkwriteisc_c(vect, name, len(name))

  end subroutine vtkwriteisc_f90

  subroutine vtkwritefsc_f90(vect, name)
    ! Wrapper routine with nicer interface.
    real(c_float), intent(in) :: vect(*)
    character(len=*), intent(in) :: name

    call vtkwritefsc_c(vect, name, len(name))

  end subroutine vtkwritefsc_f90

  subroutine vtkwritedsc_f90(vect, name)
    ! Wrapper routine with nicer interface.
    real(c_double), intent(in) :: vect(*)
    character(len=*), intent(in) :: name

    call vtkwritedsc_c(vect, name, len(name))
  end subroutine vtkwritedsc_f90

  subroutine vtkwritefvn_f90(vx, vy, vz, name)
    REAL(c_float), intent(in) :: vx(*), vy(*), vz(*)
    character(len=*) name

    call vtkwritefvn_c(vx, vy, vz, name, len(name))

  end subroutine vtkwritefvn_f90

  subroutine vtkwritedvn_f90(vx, vy, vz, name)
    REAL(c_double), intent(in) :: vx(*), vy(*), vz(*)
    character(len=*) name

    call vtkwritedvn_c(vx, vy, vz, name, len(name))

  end subroutine vtkwritedvn_f90

  subroutine vtkwritefvc_f90(vx, vy, vz, name)
    REAL(c_float), intent(in) :: vx(*), vy(*), vz(*)
    character(len=*) name

    call vtkwritefvc_c(vx, vy, vz, name, len(name))

  end subroutine vtkwritefvc_f90

  subroutine vtkwritedvc_f90(vx, vy, vz, name)
    REAL(c_double), intent(in) :: vx(*), vy(*), vz(*)
    character(len=*) name

    call vtkwritedvc_c(vx, vy, vz, name, len(name))

  end subroutine vtkwritedvc_f90

  subroutine vtkwriteftn_f90(v1, v2, v3, v4, v5, v6, v7, v8, v9, name)
    REAL(c_float), intent(in) :: v1(*), v2(*), v3(*), v4(*), v5(*), v6(*), v7(*), v8(*), v9(*)
    character(len=*) name

    call vtkwriteftn_c(v1, v2, v3, v4, v5, v6, v7, v8, v9, name, len(name))

  end subroutine vtkwriteftn_f90

  subroutine vtkwritedtn_f90(v1, v2, v3, v4, v5, v6, v7, v8, v9, name)
    REAL(c_double), intent(in) :: v1(*), v2(*), v3(*), v4(*), v5(*), v6(*), v7(*), v8(*), v9(*)
    character(len=*) name

    call vtkwritedtn_c(v1, v2, v3, v4, v5, v6, v7, v8, v9, name, len(name))

  end subroutine vtkwritedtn_f90

  subroutine vtkwriteftc_f90(v1, v2, v3, v4, v5, v6, v7, v8, v9, name)
    REAL(c_float), intent(in) :: v1(*), v2(*), v3(*), v4(*), v5(*), v6(*), v7(*), v8(*), v9(*)
    character(len=*) name

    call vtkwriteftc_c(v1, v2, v3, v4, v5, v6, v7, v8, v9, name, len(name))

  end subroutine vtkwriteftc_f90

  subroutine vtkwritedtc_f90(v1, v2, v3, v4, v5, v6, v7, v8, v9, name)
    REAL(c_double), intent(in) :: v1(*), v2(*), v3(*), v4(*), v5(*), v6(*), v7(*), v8(*), v9(*)
    character(len=*) name

    call vtkwritedtc_c(v1, v2, v3, v4, v5, v6, v7, v8, v9, name, len(name))

  end subroutine vtkwritedtc_f90

  subroutine vtksetactivescalars_f90(name)
    character(len=*) name

    call vtksetactivescalars_c(name, len_trim(name))

  end subroutine vtksetactivescalars_f90

  subroutine vtksetactivevectors_f90(name)
    character(len=*) name

    call vtksetactivevectors_c(name, len_trim(name))

  end subroutine vtksetactivevectors_f90

  subroutine vtksetactivetensors_f90(name)
    character(len=*) name

    call vtksetactivetensors_c(name, len_trim(name))

  end subroutine vtksetactivetensors_f90

end module vtkfortran
