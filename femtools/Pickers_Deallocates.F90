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

module pickers_deallocates

  use fldebug
  use fields_data_types
  use global_parameters, only : empty_name
  use picker_data_types
  use pickers_base
  use reference_counting

  implicit none

  private

  public :: deallocate, nullify, remove_picker, addref, incref, has_references

  interface deallocate
    module procedure deallocate_picker
  end interface deallocate

  interface nullify
    module procedure nullify_picker
  end interface nullify

  interface node_owner_finder_reset
    subroutine cnode_owner_finder_reset(id)
      implicit none
      integer, intent(in) :: id
    end subroutine cnode_owner_finder_reset
  end interface node_owner_finder_reset

#include "Reference_count_interface_picker_type.F90"

contains

#include "Reference_count_picker_type.F90"

  subroutine deallocate_picker(picker)
    !!< Deallocate a picker

    type(picker_type), intent(inout) :: picker

    call decref(picker)
    if(has_references(picker)) return

    ewrite(2, *) "Deallocating picker with ID", picker%picker_id
    call node_owner_finder_reset(picker%picker_id)
    call nullify(picker)

  end subroutine deallocate_picker

  subroutine nullify_picker(picker)
    !!< Return a picker type to its uninitialised state

    type(picker_type), intent(inout) :: picker

    type(picker_type) :: null_picker

    ! Initialise the null_picker name to prevent uninitialised variable access
    call set_picker_name(picker, empty_name)
    picker = null_picker

  end subroutine nullify_picker

  subroutine remove_picker(field)
    !!< Remove the picker from the supplied Coordinate field

    type(vector_field), intent(inout) :: field

    assert(associated(field%picker))
    if(associated(field%picker%ptr)) then
       call deallocate(field%picker%ptr)
       deallocate(field%picker%ptr)
       nullify(field%picker%ptr)
    end if

  end subroutine remove_picker

end module pickers_deallocates
