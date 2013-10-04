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
module sparsity_patterns_meshes
  !! Calculate shape functions and sparsity patterns.
  use sparse_tools
  use sparsity_patterns
  use shape_functions
  use fields
  use state_module
  use fldebug
  use global_parameters, only : FIELD_NAME_LEN

  implicit none

  private
  public :: get_csr_sparsity_firstorder, get_csr_sparsity_secondorder, &
       & get_csr_sparsity_compactdgdouble

  interface get_csr_sparsity_firstorder
    module procedure get_csr_sparsity_firstorder_single_state, get_csr_sparsity_firstorder_multiple_states
  end interface

  interface get_csr_sparsity_secondorder
    module procedure get_csr_sparsity_secondorder_single_state, get_csr_sparsity_secondorder_multiple_states
  end interface

  interface get_csr_sparsity_compactdgdouble
    module procedure get_csr_sparsity_compactdgdouble_single_state, get_csr_sparsity_compactdgdouble_multiple_states
  end interface

contains

  function get_csr_sparsity_firstorder_single_state(state, rowmesh, colmesh) result(sparsity)
    !!< Tries to extract a first order csr_sparsity from the supplied state using the name
    !!< formula rowmesh%name//colmesh%name//Sparsity
    !!<
    !!< If unsuccesful it creates that sparsity and inserts it into state
    !!< before returning a pointer to the newly created sparsity.
    type(csr_sparsity), pointer :: sparsity
    type(state_type), intent(inout) :: state
    type(mesh_type), intent(inout) :: rowmesh, colmesh

    type(state_type), dimension(1) :: states

    states = (/state/)
    sparsity=>get_csr_sparsity_firstorder(states, rowmesh, colmesh)
    state = states(1)

  end function get_csr_sparsity_firstorder_single_state

  function get_csr_sparsity_firstorder_multiple_states(states, rowmesh, colmesh) result(sparsity)
    !!< Tries to extract a first order csr_sparsity from any of the supplied states using the name
    !!< formula rowmesh%name//colmesh%name//Sparsity
    !!<
    !!< If unsuccesful it creates that sparsity and inserts (and aliases) it into states
    !!< before returning a pointer to the newly created sparsity.
    type(csr_sparsity), pointer :: sparsity
    type(state_type), dimension(:), intent(inout) :: states
    type(mesh_type), intent(inout) :: rowmesh, colmesh

    integer :: stat
    character(len=FIELD_NAME_LEN) :: name
    type(csr_sparsity) :: temp_sparsity

    name = trim(rowmesh%name)//trim(colmesh%name)//"Sparsity"

    sparsity => extract_csr_sparsity(states, trim(name), stat)

    if(stat/=0) then
      temp_sparsity=make_sparsity(rowmesh, colmesh, trim(name))
      call insert(states, temp_sparsity, trim(name))
      call deallocate(temp_sparsity)

      sparsity => extract_csr_sparsity(states, trim(name))
    end if

  end function get_csr_sparsity_firstorder_multiple_states

  function get_csr_sparsity_secondorder_single_state(state, rowmesh, colmesh) result(sparsity)
    !!< Tries to extract a second order csr_sparsity from the supplied state using the name
    !!< formula rowmesh%name//colmesh%name//DoubleSparsity
    !!<
    !!< If unsuccesful it creates that sparsity and inserts it into state
    !!< before returning a pointer to the newly created sparsity.
    type(csr_sparsity), pointer :: sparsity
    type(state_type), intent(inout) :: state
    type(mesh_type), intent(inout) :: rowmesh, colmesh

    type(state_type), dimension(1) :: states

    states = (/state/)
    sparsity=>get_csr_sparsity_secondorder(states, rowmesh, colmesh)
    state = states(1)

  end function get_csr_sparsity_secondorder_single_state

  function get_csr_sparsity_secondorder_multiple_states(states, rowmesh, colmesh) result(sparsity)
    !!< Tries to extract a second order csr_sparsity from any of the supplied states using the name
    !!< formula rowmesh%name//colmesh%name//DoubleSparsity
    !!<
    !!< If unsuccesful it creates that sparsity and inserts (and aliases) it into states
    !!< before returning a pointer to the newly created sparsity.
    type(csr_sparsity), pointer :: sparsity
    type(state_type), dimension(:), intent(inout) :: states
    type(mesh_type), intent(inout) :: rowmesh, colmesh

    integer :: stat
    character(len=FIELD_NAME_LEN) :: name
    type(csr_sparsity) :: temp_sparsity

    name = trim(rowmesh%name)//trim(colmesh%name)//"DoubleSparsity"

    sparsity => extract_csr_sparsity(states, trim(name), stat)

    if(stat/=0) then
      temp_sparsity=make_sparsity_transpose(rowmesh, colmesh, trim(name))
      call insert(states, temp_sparsity, trim(name))
      call deallocate(temp_sparsity)

      sparsity => extract_csr_sparsity(states, trim(name))
    end if

  end function get_csr_sparsity_secondorder_multiple_states

  function get_csr_sparsity_compactdgdouble_single_state(state, mesh) &
       & result(sparsity)
    !!< Tries to extract a compactdgdouble
    !!< csr_sparsity from the supplied state using the name
    !!< formula rowmesh%name//colmesh%name//CompactDGDoubleSparsity
    !!<
    !!< If unsuccesful it creates that sparsity and inserts it into state
    !!< before returning a pointer to the newly created sparsity.
    type(csr_sparsity), pointer :: sparsity
    type(state_type), intent(inout) :: state
    type(mesh_type), intent(inout) :: mesh

    type(state_type), dimension(1) :: states

    states = (/state/)
    sparsity=>get_csr_sparsity_compactdgdouble(states, mesh)
    state = states(1)

  end function get_csr_sparsity_compactdgdouble_single_state

  function get_csr_sparsity_compactdgdouble_multiple_states(states, mesh) &
       & result(sparsity)
    !!< Tries to extract a compactdgdouble
    !!< csr_sparsity from the supplied state using the name
    !!< formula mesh%name//CompactDGDoubleSparsity
    !!<
    !!< If unsuccesful it creates that sparsity and inserts
    !!< (and aliases) it into states
    !!< before returning a pointer to the newly created sparsity.
    type(csr_sparsity), pointer :: sparsity
    type(state_type), dimension(:), intent(inout) :: states
    type(mesh_type), intent(inout) :: mesh

    integer :: stat
    character(len=FIELD_NAME_LEN) :: name
    type(csr_sparsity) :: temp_sparsity

    name = trim(mesh%name)//"CompactDGDoubleSparsity"

    sparsity => extract_csr_sparsity(states, trim(name), stat)

    if(stat/=0) then
      temp_sparsity=make_sparsity_compactdgdouble(mesh, trim(name))
      call insert(states, temp_sparsity, trim(name))
      call deallocate(temp_sparsity)

      sparsity => extract_csr_sparsity(states, trim(name))
    end if

  end function get_csr_sparsity_compactdgdouble_multiple_states

end module sparsity_patterns_meshes
