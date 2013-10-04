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
module reference_counting
  !! A module to implement reference counting on fields.
  use fldebug
  use global_parameters, only: FIELD_NAME_LEN, current_debug_level
  implicit none

  private

  type refcount_type
     !!< Type to hold reference count for an arbitrary object.
     type(refcount_type), pointer :: prev=>null(), next=>null()
     integer :: count=0
     integer :: id
     character(len=FIELD_NAME_LEN) :: name
     character(len=FIELD_NAME_LEN) :: type
     logical :: tagged=.false.
  end type refcount_type

  ! Linked lists to track fields to which references exist.
  type(refcount_type), save, target :: refcount_list

  public print_references, refcount_type, &
         refcount_list, new_refcount, &
         tag_references, print_tagged_references

contains

  function new_refcount(type, name)
    !! Allocate a new refcount and place it in the refcount_list.
    type(refcount_type), pointer :: new_refcount
    character(len=*), intent(in) :: type, name


    allocate(new_refcount)

    new_refcount%count=1

    new_refcount%name=name
    new_refcount%type=type
    new_refcount%tagged=.false. ! just to be sure

    ! Add the new refcounter at the head of the list.
    new_refcount%next=>refcount_list%next
    refcount_list%next=>new_refcount

    ! Reverse pointers
    new_refcount%prev=>refcount_list
    if (associated(new_refcount%next)) then
       new_refcount%next%prev=>new_refcount
    end if

  end function new_refcount

  subroutine print_references(priority)
    !!< Print out a list of currently allocated fields and their reference
    !!< counts. This results in ewrites with the given priority.
    integer, intent(in) :: priority

    type(refcount_type), pointer :: this_ref

    ! the first 2 ewrites have fixed priority, so we can call print_references
    ! with priority 0 to print warnings *only* if there are any references left.
    ewrite(1,*) "Printing out all currently allocated references:"
    this_ref=>refcount_list%next
    if (.not.associated(this_ref)) then
       ewrite(1,*) "There are no references left."
    end if
    do
       if (.not.associated(this_ref)) then
          exit
       end if

       ewrite(priority, '(a,i0)') " " // trim(this_ref%type)//&
            " " // trim(this_ref%name)//&
            " has reference count ", this_ref%count, &
            " and id ", this_ref%id

       this_ref=>this_ref%next
    end do

  end subroutine print_references

  subroutine tag_references
    !!< Tags all current references, so they can later be printed with
    !!< print_tagged_references. This can be used if all current objects are
    !!< planned for deallocation, but not before new objects are allocated.
    !!< The newly allocated objects will not be tagged and therefore after
    !!< we've finally deallocated the objects we planned to deallocate,
    !!< we can check whether all references have gone
    !!< with print_tagged_references without printing any new references.

    type(refcount_type), pointer :: this_ref

    this_ref=>refcount_list%next
    if (.not.associated(this_ref)) return ! no references yet/left

    do
       if (.not.associated(this_ref)) exit
       this_ref%tagged=.true.
       this_ref=>this_ref%next
    end do

  end subroutine tag_references

  subroutine print_tagged_references(priority)
    !!< Print out a list of all objects
    !!< that have been allocated before the last call to tag_references.
    !!< This results in ewrites with the given priority.
    integer, intent(in) :: priority

    type(refcount_type), pointer :: this_ref
    logical no_tags

    ! the first 2 ewrites have fixed priority, so we can call print_references
    ! with priority 0 to print warnings *only* if there are any references left.
    ewrite(1,*) "Printing out all tagged references:"
    this_ref=>refcount_list%next
    if (.not.associated(this_ref)) then
       ewrite(1,*) "There are no tagged references left."
    end if

    no_tags=.true.
    do
       if (.not.associated(this_ref)) then
          exit
       end if

       if (this_ref%tagged) then

          ewrite(priority, '(a,i0)') " " // trim(this_ref%type)//&
            " " // trim(this_ref%name)//&
            " has reference count ", this_ref%count, &
            " and id ", this_ref%id
          no_tags=.false.

       end if

       this_ref=>this_ref%next
    end do

    if (no_tags) then
       ewrite(1,*) "No tagged references left."
    end if

  end subroutine print_tagged_references

end module reference_counting
