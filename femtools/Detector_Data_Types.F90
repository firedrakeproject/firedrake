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

module detector_data_types

  use fldebug
  use global_parameters, only : FIELD_NAME_LEN

  implicit none

  private

  public :: detector_type, rk_gs_parameters, detector_linked_list, &
            detector_list_ptr, stringlist, &
            STATIC_DETECTOR, LAGRANGIAN_DETECTOR

  integer, parameter :: STATIC_DETECTOR=1, LAGRANGIAN_DETECTOR=2

  type stringlist
     !!< Container type for a list of strings.
     character(len=FIELD_NAME_LEN), dimension(:), pointer :: ptr
  end type stringlist

  !! Type for caching detector position and search information.
  type detector_type
     !! Physical location of the detector.
     real, dimension(:), allocatable :: position
     !! Name of the detector in input and output.
     character(len=FIELD_NAME_LEN) :: name
     !! Element number in which the detector lies.
     integer :: element
     !! Local coordinates of the detector in that element.
     real, dimension(:), allocatable :: local_coords
     !! Whether the detector is static or Lagrangian.
     integer :: type = STATIC_DETECTOR
     !! Identification number indicating the order in which the detectors are read
     integer :: id_number
     !! ID of the parent list, needed for Zoltan to map the detector back
     integer :: list_id
     !! RK timestepping stages (first index is stage no., second index is dim)
     real, dimension(:,:), allocatable :: k
     !! RK update destination vector (size dim)
     real, dimension(:), allocatable :: update_vector
     !! Have we completed the search?
     logical :: search_complete
     !! Pointers for detector linked lists
     TYPE (detector_type), POINTER :: next=> null()
     TYPE (detector_type), POINTER :: previous=> null()
  end type detector_type

  ! Parameters for lagrangian detector movement
  type rk_gs_parameters
    ! Runk-Kutta Guided Search parameters
    integer :: n_stages, n_subcycles
    real, allocatable, dimension(:) :: timestep_weights
    real, allocatable, dimension(:,:) :: stage_matrix
    real :: search_tolerance
  end type rk_gs_parameters

  type detector_linked_list
     !! Doubly linked list implementation
     integer :: length=0
     TYPE (detector_type), pointer :: first => null()
     TYPE (detector_type), pointer :: last => null()

     !! Internal ID used for packing/unpacking detectors
     integer :: id  ! IDs are counted from 1

     !! Parameters for lagrangian movement (n_stages, stage_matrix, etc)
     type(rk_gs_parameters), pointer :: move_parameters => null()
     logical :: move_with_mesh = .false.

     !! Optional array for detector names; names are held in read order
     character(len = FIELD_NAME_LEN), dimension(:), allocatable :: detector_names

     !! List of scalar/vector fields to include in detector output
     type(stringlist), dimension(:), allocatable :: sfield_list
     type(stringlist), dimension(:), allocatable :: vfield_list
     integer :: num_sfields = 0   ! Total number of scalar fields across all phases
     integer :: num_vfields = 0   ! Total number of vector fields across all phases

     !! I/O parameters
     logical :: binary_output = .false.
     logical :: write_nan_outside = .false.
     integer :: output_unit = 0          ! Assumed non-opened as long this is 0
     integer :: mpi_fh = 0               ! MPI filehandle
     integer :: mpi_write_count = 0      ! Offset in MPI file
     integer :: total_num_det = 0        ! Global number of detectors in this list
  end type detector_linked_list

  type detector_list_ptr
     type(detector_linked_list), pointer :: ptr
  end type detector_list_ptr

end module detector_data_types
