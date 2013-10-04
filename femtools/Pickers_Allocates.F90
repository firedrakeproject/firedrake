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

#include "fdebug.h"

module pickers_allocates

  use elements
  use eventcounter
  use fields_base
  use fields_data_types
  use fldebug
  use global_parameters, only : empty_name
  use node_owner_finder
  use picker_data_types
  use pickers_base
  use pickers_deallocates

  implicit none

  private

  public :: allocate, initialise_picker, incref, has_references

  interface allocate
    module procedure allocate_picker
  end interface allocate

contains

  subroutine allocate_picker(picker, positions, name)
    !!< Allocate a picker

    type(picker_type), intent(out) :: picker
    type(vector_field), intent(in) :: positions
    character(len = *), optional, intent(in) :: name

    call node_owner_finder_set_input(picker%picker_id, positions)
    ewrite(2, *) "New picker ID: ", picker%picker_id

    if(present(name)) then
      call set_picker_name(picker, name)
    else
      call set_picker_name(picker, empty_name)
    end if

    picker%last_mesh_movement = eventcount(EVENT_MESH_MOVEMENT)

    call addref(picker)

  end subroutine allocate_picker

  subroutine initialise_picker(positions)
    !!< Initialise a picker for a Coordinate field

    type(vector_field), intent(inout) :: positions

    if(use_cached_picker(positions)) return

    ewrite(2, *) "Initialising picker for field " // trim(positions%name)
    assert(associated(positions%picker))
    if(associated(positions%picker%ptr)) call remove_picker(positions)
    allocate(positions%picker%ptr)
    call allocate(positions%picker%ptr, positions, name = trim(positions%name) // "Picker")

  contains

    function use_cached_picker(positions)
      type(vector_field), intent(in) :: positions

      logical :: use_cached_picker

      assert(associated(positions%picker))
      if(associated(positions%picker%ptr)) then
        if(eventcount(EVENT_MESH_MOVEMENT) > positions%picker%ptr%last_mesh_movement) then
          ! Mesh movement event has occurred - generate a new picker
          use_cached_picker = .false.
        else
          use_cached_picker = .true.
        end if
      else
        use_cached_picker = .false.
      end if

    end function use_cached_picker

  end subroutine initialise_picker

end module pickers_allocates
