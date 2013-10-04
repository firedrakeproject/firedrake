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

! This module contains code and types variables common to all the ExodusII I/O routinesmodule

module exodusii_common

  type EXOnode
     integer :: nodeID
     double precision :: x(3)
  end type EXOnode

  type EXOelement
     integer :: elementID, type, numTags, blockID
     integer, pointer :: tags(:), nodeIDs(:)
  end type EXOelement

  contains

  ! -----------------------------------------------------------------
  ! Tries valid exodusii file extensions and quits if none of them
  ! has been found aka file does not exist
  subroutine get_exodusii_filename(filename, lfilename, fileExists)
    character(len=*), intent(in) :: filename
    character(len=*), intent(inout) :: lfilename
    logical, intent(inout) :: fileExists
    ! An ExodusII file can have the following file extensions:
    ! e, exo, E, EXO, our first guess shall be exo
    lfilename = trim(filename)//".exo"
    inquire(file = trim(lfilename), exist = fileExists)
    if(.not. fileExists) then
      lfilename = trim(filename) // ".e"
      inquire(file = trim(lfilename), exist = fileExists)
      if(.not. fileExists) then
        lfilename = trim(filename) // ".EXO"
        inquire(file = trim(lfilename), exist = fileExists)
        if(.not. fileExists) then
          lfilename = trim(filename) // ".E"
          inquire(file = trim(lfilename), exist = fileExists)
        end if
      end if
    end if
    lfilename = trim(lfilename)
  end subroutine get_exodusii_filename



  ! -----------------------------------------------------------------
  ! Reorder to Fluidity node ordering
  subroutine toFluidityElementNodeOrdering( ele_nodes, elemType )
    integer, dimension(:), intent(inout) :: ele_nodes
    integer, intent(in) :: elemType

    integer i
    integer, dimension(size(ele_nodes)) :: nodeOrder

    ! Specify node ordering
    select case( elemType )
    ! Quads
    case (3)
       nodeOrder = (/1, 2, 4, 3/)
    ! Hexahedron
    case (5)
       nodeOrder = (/1, 2, 4, 3, 5, 6, 8, 7/)
    case default
       do i=1, size(ele_nodes)
          nodeOrder(i) = i
       end do
    end select

    ele_nodes = ele_nodes(nodeOrder)

  end subroutine toFluidityElementNodeOrdering


end module exodusii_common
