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

#include "fdebug.h"

module halo_data_types

  use futils
  use global_parameters, only : FIELD_NAME_LEN
  use mpi_interfaces
  use reference_counting
  use iso_c_binding

  implicit none

  private

  public :: halo_type, halo_pointer

  !! Halo data types
  integer, parameter, public :: HALO_TYPE_CG_NODE = 1,&
       HALO_TYPE_DG_NODE = 2, &
       HALO_TYPE_ELEMENT = 3

  !! Halo ordering schemes
  integer, parameter, public :: HALO_ORDER_GENERAL = 1, &
    & HALO_ORDER_TRAILING_RECEIVES = 2

  !! Halo information type
  type halo_type
    !! Name of this halo
    character(len = FIELD_NAME_LEN) :: name
    !! Reference count for halo
    type(refcount_type), pointer :: refcount => null()

    !! Halo data type
    integer :: data_type = 0
    !! Ordering scheme for halo
    integer :: ordering_scheme = 0

    !! The MPI communicator for this halo
#ifdef HAVE_MPI
    integer :: communicator
#else
    integer :: communicator = -1
#endif
    !! The number of processes
    integer :: nprocs = 0
    !! The sends
    type(c_ptr) :: sends_c
    type(c_ptr) :: receives_c
    type(integer_vector), dimension(:), pointer :: sends => null()
    !! The receives
    type(integer_vector), dimension(:), pointer :: receives => null()

    !! The number of owned nodes
    integer :: nowned_nodes = -1

    !! Ownership cache
    integer, dimension(:), pointer :: owners => null()

    ! Global to universal numbering mapping cache
    !! Universal number of nodes
    integer :: unn_count = -1
    !! Base for owned nodes universal node numbering of each process:
    integer, dimension(:), pointer :: owned_nodes_unn_base => null()
    !! Base for owned nodes universal node numbering for this process
    !! should be the same as owned_nodes_unn_base(rank+1):
    integer :: my_owned_nodes_unn_base = -1

    !! Map from global to universal numbers for receives
    integer, dimension(:), pointer :: receives_gnn_to_unn => null()
    type(c_ptr) :: receives_gnn_to_unn_c

    !! Map from global to universal node numbers for all items.
    !! This is required for halos which are not ordered by ownership.
    integer, dimension(:), pointer :: gnn_to_unn => null()
  end type halo_type

  type halo_pointer
    !!< Dummy type to allow for arrays of pointers to halos
    type(halo_type), pointer :: ptr => null()
  end type halo_pointer

end module halo_data_types
