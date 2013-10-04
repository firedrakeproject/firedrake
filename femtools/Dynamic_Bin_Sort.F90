!     Copyright (C) 2006 Imperial College London and others.
!
!     Please see the AUTHORS file in the main source directory for a full list
!     of copyright holders.
!
!     Prof. C Pain
!     Applied Modelling and Computation Group
!     Department of Earth Science and Engineering
!     Imperial College London
!
!     amcgsoftware@imperial.ac.uk
!
!     This library is free software; you can redistribute it and/or
!     modify it under the terms of the GNU Lesser General Public
!     License as published by the Free Software Foundation,
!     version 2.1 of the License.
!
!     This library is distributed in the hope that it will be useful,
!     but WITHOUT ANY WARRANTY; without even the implied warranty of
!     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!     Lesser General Public License for more details.
!
!     You should have received a copy of the GNU Lesser General Public
!     License along with this library; if not, write to the Free Software
!     Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
!     USA

#include "fdebug.h"

! This module contains the code for doing "dynamic" bin sorts
! where the elements change bin during the sort. Possible applications
! are in element/node orderings (e.g. minimum degree, verticaldg ordering)

module dynamic_bin_sort_module
use fldebug
implicit none

  type, public:: dynamic_bin_type
    ! start of each bin in bin(:) array
    integer, dimension(:), pointer:: start
    ! size of each bin in bin(:) array
    ! (note that we allow for padding between bins)
    integer, dimension(:), pointer:: size
    ! the bin(:) array containing the consecutive bins
    integer, dimension(:), pointer:: bin
    ! for each element the index in bin(:) where it is located
    ! i.e. a reverse map bin(bin_index(i))=i
    integer, dimension(:), pointer:: index
    ! bin number the element is stored in:
    integer, dimension(:), pointer:: bin_no
  end type dynamic_bin_type

  interface allocate
     module procedure allocate_dynamic_bins
  end interface allocate

  interface deallocate
     module procedure deallocate_dynamic_bins
  end interface deallocate

  public allocate, deallocate, move_element, pull_element, pull_from_bin, &
     element_pulled

contains

  subroutine allocate_dynamic_bins(dbin, binlist)
  !!< Allocate and initialise the dynamic bins with a binlist
  !!< that gives the bin number that each element is in.
  !!< #elements=size(binlist), bins are numbered 1:maxval(binlist)
  !!< The a pointer to binlist is stored and binlist is updated if
  !!< elements are moved to a different bin with move_element().
  type(dynamic_bin_type), intent(out):: dbin
  integer, dimension(:), target, intent(in):: binlist

    integer nbins, nelements
    integer i, k, new_index

    nelements=size(binlist)
    nbins=maxval(binlist)

    allocate(dbin%bin(1:nelements), dbin%index(1:nelements), &
             dbin%start(1:nbins+1), &
             dbin%size(1:nbins) )
    dbin%bin_no => binlist

    ! count n/o elements per bin
    dbin%size=0
    do i=1, nelements
      dbin%size(binlist(i))=dbin%size(binlist(i))+1
    end do

    assert(sum(dbin%size)==nelements)

    ! work out the starting point of each bin:
    k=1
    do i=1, nbins
      dbin%start(i)=k
      k=k+dbin%size(i)
    end do
    dbin%start(nbins+1)=k

    ! set bin size back to 0, so we can let it grow back as we insert the elements
    dbin%size=0
    ! insert all elements
    do i=1, nelements
      new_index=dbin%start(binlist(i))+dbin%size(binlist(i))
      dbin%bin(new_index)=i
      dbin%index(i)=new_index
      dbin%size(binlist(i))=dbin%size(binlist(i))+1
    end do

  end subroutine allocate_dynamic_bins

  subroutine deallocate_dynamic_bins(dbin)
  type(dynamic_bin_type), intent(inout):: dbin

    deallocate(dbin%bin, dbin%index, dbin%start, dbin%size)
    ! dbin%bin_no is a pointer to the binlist supplied to allocate

  end subroutine deallocate_dynamic_bins

  subroutine move_element(dbin, element, bin_no)
  !!< move an element (thas has been inserted before)
  !!< to a different bin
  type(dynamic_bin_type), intent(inout):: dbin
  integer, intent(in):: element, bin_no

    integer prev_index, prev_bin_no, last, new_index

    prev_index=dbin%index(element)
    ! this routine should only be used to move elements that have been inserted already:
    assert(prev_index/=0)

    prev_bin_no=dbin%bin_no(element)
    ! just to make sure:
    if (prev_bin_no==bin_no) return

    ! remove the element from its previous bin by overwriting with
    ! the last element of that bin
    last=dbin%bin(dbin%start(prev_bin_no)+dbin%size(prev_bin_no)-1)
    dbin%index(last)=prev_index
    dbin%bin(prev_index)=last
    dbin%size(prev_bin_no)=dbin%size(prev_bin_no)-1

    if (bin_no<prev_bin_no) then
      ! insert at the end:
      new_index=dbin%start(bin_no)+dbin%size(bin_no)
      ! first check if there's still place
      if (new_index==dbin%start(bin_no+1)) then
        ! if not make room for it
        call shuffle_bin_right(dbin, bin_no+1)
      end if

      dbin%bin(new_index)=element
      dbin%index(element)=new_index
      dbin%size(bin_no)=dbin%size(bin_no)+1

    else

      ! insert at the beginning:
      new_index=dbin%start(bin_no)-1
      ! first check if there's still place
      if (new_index<dbin%start(bin_no-1)+dbin%size(bin_no-1)) then
        ! if not make room for it
        call shuffle_bin_left(dbin, bin_no-1)
      end if

      dbin%bin(new_index)=element
      dbin%index(element)=new_index
      dbin%start(bin_no)=new_index
      dbin%size(bin_no)=dbin%size(bin_no)+1

    end if

    dbin%bin_no(element)=bin_no

  end subroutine move_element

  recursive subroutine shuffle_bin_left(dbin, bin_no)
  ! shuffle the specified bin to the left
  ! (and those to the left of it if needed)
  type(dynamic_bin_type), intent(inout):: dbin
  integer, intent(in):: bin_no

    integer last

    if (bin_no==1) then
      ewrite(0,*) "No room to the left"
      FLAbort("Something went wrong in dynamic bin sort algorithm")
    end if

    if (dbin%start(bin_no-1)+dbin%size(bin_no-1)==dbin%start(bin_no)) then
      call shuffle_bin_left(dbin, bin_no-1)
    end if

    ! move last element to first position
    last=dbin%bin(dbin%start(bin_no)+dbin%size(bin_no)-1)
    dbin%start(bin_no)=dbin%start(bin_no)-1
    dbin%index(last)=dbin%start(bin_no)
    dbin%bin(dbin%start(bin_no))=last

  end subroutine shuffle_bin_left

  recursive subroutine shuffle_bin_right(dbin, bin_no)
  ! shuffle the specified bin to the right
  ! (and those to the right of it if needed)
  type(dynamic_bin_type), intent(inout):: dbin
  integer, intent(in):: bin_no

    integer first

    if (bin_no==size(dbin%start)) then
      ewrite(0,*) "No room to the right"
      FLAbort("Something went wrong in dynamic bin sort algorithm")
    end if

    if (dbin%start(bin_no)+dbin%size(bin_no)==dbin%start(bin_no+1)) then
      call shuffle_bin_right(dbin, bin_no+1)
    end if

    ! move first element to last position
    first=dbin%bin(dbin%start(bin_no))
    dbin%index(first)=dbin%start(bin_no)+dbin%size(bin_no)
    dbin%bin(dbin%start(bin_no)+dbin%size(bin_no))=first
    dbin%start(bin_no)=dbin%start(bin_no)+1

  end subroutine shuffle_bin_right

  subroutine pull_element(dbin, element, bin_no)
  !!< pull an element from the first non-empty bin
  type(dynamic_bin_type), intent(inout):: dbin
  integer, intent(out):: element
  ! the bin it's pulled from:
  integer, intent(out):: bin_no

    integer i

    ! find first non-empty bin:
    do i=1, size(dbin%size)
      if (dbin%size(i)>0) exit
    end do
    if (i>size(dbin%size)) then
      FLAbort("Tried to pull an element while all bins are empty.")
    end if
    bin_no=i

    call pull_from_bin(dbin, bin_no, element)

  end subroutine pull_element

  subroutine pull_from_bin(dbin, bin_no, element)
  !!< pull an element from the specified bin
  type(dynamic_bin_type), intent(inout):: dbin
  integer, intent(in):: bin_no
  integer, intent(out):: element

    integer index

    index=dbin%start(bin_no)+dbin%size(bin_no)-1
    element=dbin%bin(index)

    ! remove the element:
    dbin%index(element)=0
    dbin%size(bin_no)=dbin%size(bin_no)-1

  end subroutine pull_from_bin

  logical function element_pulled(dbin, element)
  !!< whether an element has been pulled already
  type(dynamic_bin_type), intent(in):: dbin
  integer, intent(in):: element

    element_pulled= dbin%index(element)==0

  end function element_pulled

end module dynamic_bin_sort_module
