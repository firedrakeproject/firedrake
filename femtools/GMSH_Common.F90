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
!    License as published by the Free Software Foundation; either
!    version 2.1 of the License, or (at your option) any later version.
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



! This module contains code and variables common to all the GMSH I/O routines


module gmsh_common

  character(len=3), parameter :: GMSHVersionStr = "2.1"
  integer, parameter :: asciiFormat = 0
  integer, parameter :: binaryFormat = 1
  ! Anyway to automatically calc this in Fortran?
  integer, parameter :: doubleNumBytes = 8

  integer, parameter :: longStringLen = 1000
  real, parameter :: verySmall = 10e-10

  ! For each type, the number of nodes. -1 means unsupported
  integer, dimension(15) :: elementNumNodes = (/ &
       2, 3, 4, 4, 8, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /)

  type GMSHnode
     integer :: nodeID, columnID
     double precision :: x(3)
     ! Currently unused
     ! real, pointer :: properties(:)
  end type GMSHnode

  type GMSHelement
     integer :: elementID, type, numTags
     integer, pointer :: tags(:), nodeIDs(:)
  end type GMSHelement


contains

  ! -----------------------------------------------------------------
  ! Change already-open file to ASCII formatting
  ! Involves a bit of sneaky code.

  subroutine ascii_formatting(fd, filename, readWriteStr)
    integer fd
    character(len=*) :: filename, readWriteStr

    integer position


    inquire(fd, POS=position)
    close(fd)

    select case( trim(readWriteStr) )
    case("read")
#ifdef __INTEL_COMPILER
#if __INTEL_COMPILER < 1211
    ! Gets around a bug described in:
    ! http://software.intel.com/en-us/forums/showthread.php?t=101333
       position = position-1
#endif
#endif
       open( fd, file=trim(filename), action="read", form="formatted", &
            access="stream")
       read( fd, "(I1)", POS=position, ADVANCE="no" )

    case("write")
       open( fd, file=trim(filename), action="write",  form="formatted", &
            access="stream", position="append")

    end select


  end subroutine ascii_formatting


  ! -----------------------------------------------------------------
  ! Change already-open file to binary formatting
  ! Sneaky code, as above.

  subroutine binary_formatting(fd, filename, readWriteStr)
    integer fd
    character(len=*) filename, readWriteStr

    integer position


    inquire(fd, POS=position)
    close(fd)

    select case( trim(readWriteStr) )
    case("read")
       open( fd, file=trim(filename), action="read", form="unformatted", &
            access="stream")
       read( fd, POS=position )

    case("write")
       open( fd, file=trim(filename), action="write", form="unformatted", &
            access="stream", position="append")

    end select

  end subroutine binary_formatting


  ! -----------------------------------------------------------------
  ! Reorder to Fluidity node ordering

  subroutine toFluidityElementNodeOrdering( oldList, elemType )
    integer, pointer :: oldList(:), flNodeList(:), nodeOrder(:)
    integer i, elemType

    numNodes = size(oldList)
    allocate( flNodeList(numNodes) )
    allocate( nodeOrder(numNodes) )

    ! Specify node ordering
    select case( elemType )
    ! Quads
    case (3)
       nodeOrder = (/1, 2, 4, 3/)
    ! Hexahedron
    case (5)
       nodeOrder = (/1, 2, 4, 3, 5, 6, 8, 7/)
    case default
       do i=1, numNodes
          nodeOrder(i) = i
       end do
    end select

    ! Reorder nodes
    do i=1, numNodes
       flNodeList(i) = oldList( nodeOrder(i) )
    end do

    ! Allocate to original list, and dealloc temp list.
    oldList(:) = flNodeList(:)
    deallocate( flNodeList )
    !deallocate(nodeOrder)

  end subroutine toFluidityElementNodeOrdering



  ! -----------------------------------------------------------------
  ! Reorder Fluidity node ordering to GMSH

  subroutine toGMSHElementNodeOrdering( oldList, elemType )
    integer, pointer :: oldList(:), gmshNodeList(:), nodeOrder(:)
    integer i, elemType


    numNodes = size(oldList)
    allocate( gmshNodeList(numNodes) )
    allocate( nodeOrder(numNodes) )

    ! Specify node ordering
    select case( elemType )
    ! Quads
    case (3)
       nodeOrder = (/1, 2, 4, 3/)
    ! Hexahedron
    case (5)
       nodeOrder = (/1, 2, 4, 3, 5, 6, 8, 7/)

    case default
       do i=1, numNodes
          nodeOrder(i) = i
       end do
    end select

    ! Reorder nodes
    do i=1, numNodes
       gmshNodeList(i) = oldList( nodeOrder(i) )
    end do

    ! Allocate to original list, and dealloc temp list.
    oldList(:) = gmshNodeList(:)

    deallocate( gmshNodeList )
    !deallocate( nodeOrder )

  end subroutine toGMSHElementNodeOrdering



  subroutine deallocateElementList( elements )
    type(GMSHelement), pointer :: elements(:)
    integer i

    do i = 1, size(elements)
       deallocate(elements(i)%tags)
       deallocate(elements(i)%nodeIDs)
    end do

    deallocate( elements )

  end subroutine deallocateElementList

end module gmsh_common
