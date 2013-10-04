!    Copyright (C) 2007 Imperial College London and others.
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

module state_module
  !!< This module provides a wrapper object which allows related groups of
  !!< fields to be passed around together.
  use global_parameters, only:OPTION_PATH_LEN, empty_path
  use fields_data_types
  use fields_allocates
  use fields_base
  use fields_manipulation
  use halo_data_types
  use halos_allocates
  use halos_communications
  use sparse_tools
  use futils, only: int2str
  use linked_lists
  implicit none

  private

  type state_type
     !!< This type allows sets of fields and meshes to be passed around
     !!< together and retrieved by name.

     !! name for the state
     character(len=FIELD_NAME_LEN) :: name =""

     !! option path for state
#ifdef DDEBUG
     character(len=OPTION_PATH_LEN) :: option_path="/uninitialised_path/"
#else
     character(len=OPTION_PATH_LEN) :: option_path
#endif

     !! The names used for fields should, where possible, be taken from the
     !! the CGNS SIDS.
     character(len=FIELD_NAME_LEN), dimension(:), pointer :: &
          vector_names=>null(), scalar_names=>null(), mesh_names=>null(), &
          halo_names=>null(), tensor_names=>null(), &
          csr_sparsity_names=>null(), csr_matrix_names=>null(), &
          block_csr_matrix_names=>null()
     type(vector_field_pointer), dimension(:), pointer :: vector_fields=>null()
     type(tensor_field_pointer), dimension(:), pointer :: tensor_fields=>null()
     type(scalar_field_pointer), dimension(:), pointer :: scalar_fields=>null()
     type(mesh_pointer), dimension(:), pointer :: meshes=>null()
     type(halo_pointer), dimension(:), pointer :: halos=>null()
     type(csr_sparsity_pointer), dimension(:), pointer :: csr_sparsities => null()
     type(csr_matrix_pointer), dimension(:), pointer :: csr_matrices=>null()
     type(block_csr_matrix_pointer), dimension(:), pointer :: block_csr_matrices=>null()
  end type state_type

  interface deallocate
     module procedure deallocate_state, deallocate_state_vector, deallocate_state_rank_2
  end interface

  interface nullify
     module procedure nullify_state
  end interface

  interface insert
     module procedure insert_tensor_field, insert_vector_field, insert_scalar_field, &
       insert_mesh, insert_halo, insert_csr_sparsity, insert_csr_matrix, insert_block_csr_matrix, &
       insert_and_alias_scalar_field, insert_and_alias_vector_field, insert_and_alias_tensor_field, &
       insert_and_alias_csr_matrix, insert_and_alias_block_csr_matrix, insert_and_alias_mesh, &
       insert_and_alias_halo, insert_and_alias_csr_sparsity
  end interface

  interface extract_scalar_field
     module procedure extract_from_one_scalar_field, extract_from_any_scalar_field, &
           & extract_scalar_field_by_index
  end interface

  interface extract_vector_field
     module procedure extract_from_one_vector_field, extract_from_any_vector_field, &
       extract_vector_field_by_index
  end interface

  interface extract_tensor_field
     module procedure extract_tensor_field, extract_tensor_field_by_index
  end interface

  interface extract_mesh
     module procedure extract_mesh_from_one, extract_mesh_from_any, extract_mesh_by_index
  end interface

  interface extract_halo
    module procedure extract_halo, extract_halo_by_index
  end interface extract_halo

  interface extract_csr_sparsity
     module procedure extract_from_one_csr_sparsity, &
           & extract_from_any_csr_sparsity, extract_csr_sparsity_by_index
  end interface

  interface extract_csr_matrix
      module procedure extract_from_one_csr_matrix,&
           & extract_from_any_csr_matrix, extract_csr_matrix_by_index
  end interface

  interface extract_block_csr_matrix
      module procedure extract_from_one_block_csr_matrix,&
           & extract_from_any_block_csr_matrix,&
           & extract_block_csr_matrix_by_index
  end interface

  interface has_halo
      module procedure state_has_halo
  end interface has_halo

  interface halo_count
      module procedure halo_count_state
  end interface halo_count

  interface collapse_state
      module procedure collapse_single_state, collapse_multiple_states
  end interface

  interface collapse_fields_in_state
      module procedure collapse_fields_in_single_state, &
        & collapse_fields_in_multiple_states
  end interface

  interface halo_update
      module procedure halo_update_state, halo_update_states
  end interface

  interface aliased
      module procedure aliased_scalar, aliased_vector, aliased_tensor
  end interface

  public state_type, deallocate, insert, nullify
  public field_rank, extract_scalar_field, extract_vector_field, extract_tensor_field
  public extract_field_mesh, extract_mesh, extract_halo
  public extract_csr_sparsity, extract_csr_matrix, extract_block_csr_matrix
  public has_scalar_field, has_vector_field, has_tensor_field, has_mesh, has_halo
  public has_csr_sparsity, has_csr_matrix, has_block_csr_matrix
  public get_state_index, print_state, select_state_by_mesh
  public remove_tensor_field, remove_vector_field, remove_scalar_field
  public remove_csr_sparsity, remove_csr_matrix, remove_block_csr_matrix
  public scalar_field_count, vector_field_count, tensor_field_count, field_count
  public mesh_count, halo_count, csr_sparsity_count, csr_matrix_count,&
       & block_csr_matrix_count
  public set_vector_field_in_state
  public collapse_state, extract_state, collapse_fields_in_state
  public set_option_path
  public unique_mesh_count, sort_states_by_mesh, halo_update
  public aliased

  !! Fields which exist only so that extract does not return null
  type(vector_field), save, target :: fake_vector_field
  type(scalar_field), save, target :: fake_scalar_field
  type(tensor_field), save, target :: fake_tensor_field
  type(mesh_type), save, target :: fake_mesh
  type(halo_type), save, target :: fake_halo
  type(csr_sparsity), save, target :: fake_csr_sparsity
  type(csr_matrix), save, target :: fake_csr_matrix
  type(block_csr_matrix), save, target :: fake_block_csr_matrix

contains

  subroutine deallocate_state(state)
    !!< Clear out all references in state. Note that since state grows to
    !!< the right size, it is neither necessary nor possible to allocate
    !!< state.
    type(state_type), intent(inout) :: state
    integer :: i

    if (associated(state%vector_names)) then
       deallocate(state%vector_names)
    end if
    if (associated(state%scalar_names)) then
       deallocate(state%scalar_names)
    end if
    if (associated(state%mesh_names)) then
       deallocate(state%mesh_names)
    end if
    if (associated(state%halo_names)) then
       deallocate(state%halo_names)
    end if
    if (associated(state%tensor_names)) then
       deallocate(state%tensor_names)
    end if
    if (associated(state%csr_sparsity_names)) then
       deallocate(state%csr_sparsity_names)
    end if
    if (associated(state%csr_matrix_names)) then
       deallocate(state%csr_matrix_names)
    end if
    if (associated(state%block_csr_matrix_names)) then
       deallocate(state%block_csr_matrix_names)
    end if
    if (associated(state%vector_fields)) then
       do i=1,size(state%vector_fields)
          call deallocate(state%vector_fields(i)%ptr)
          deallocate(state%vector_fields(i)%ptr)
       end do
       deallocate(state%vector_fields)
    end if
    if (associated(state%scalar_fields)) then
       do i=1,size(state%scalar_fields)
          call deallocate(state%scalar_fields(i)%ptr)
          deallocate(state%scalar_fields(i)%ptr)
       end do
       deallocate(state%scalar_fields)
    end if
    if (associated(state%tensor_fields)) then
       do i=1,size(state%tensor_fields)
          call deallocate(state%tensor_fields(i)%ptr)
          deallocate(state%tensor_fields(i)%ptr)
       end do
       deallocate(state%tensor_fields)
    end if
    if (associated(state%meshes)) then
       do i=1,size(state%meshes)
          call deallocate(state%meshes(i)%ptr)
          deallocate(state%meshes(i)%ptr)
       end do
       deallocate(state%meshes)
    end if
    if (associated(state%halos)) then
       do i=1,size(state%halos)
          call deallocate(state%halos(i)%ptr)
          deallocate(state%halos(i)%ptr)
       end do
    end if
    if (associated(state%csr_sparsities)) then
       do i=1,size(state%csr_sparsities)
          call deallocate(state%csr_sparsities(i)%ptr)
          deallocate(state%csr_sparsities(i)%ptr)
       end do
       deallocate(state%csr_sparsities)
    end if
    if (associated(state%csr_matrices)) then
       do i=1,size(state%csr_matrices)
          call deallocate(state%csr_matrices(i)%ptr)
          deallocate(state%csr_matrices(i)%ptr)
       end do
       deallocate(state%csr_matrices)
    end if
    if (associated(state%block_csr_matrices)) then
       do i=1,size(state%block_csr_matrices)
          call deallocate(state%block_csr_matrices(i)%ptr)
          deallocate(state%block_csr_matrices(i)%ptr)
       end do
       deallocate(state%block_csr_matrices)
    end if

  end subroutine deallocate_state

  subroutine deallocate_state_vector(state)
    type(state_type), dimension(:), intent(inout) :: state

    integer :: i

    do i = 1, size(state)
      call deallocate(state(i))
    end do

  end subroutine deallocate_state_vector

  subroutine deallocate_state_rank_2(state)
    type(state_type), dimension(:,:), intent(inout) :: state
    integer :: i, j

    do i=1,size(state, 1)
      do j=1,size(state, 2)
        call deallocate(state(i,j))
      end do
    end do
  end subroutine deallocate_state_rank_2

  elemental subroutine nullify_state(state)
    !!< Nullify all the pointers in state. This should not be necessary but
    !!< it appears there is a gfortran bug which causes array components
    !!< not to be nullified.
    type(state_type), intent(inout) :: state

    state%vector_names=>null()
    state%scalar_names=>null()
    state%mesh_names=>null()
    state%halo_names=>null()
    state%tensor_names=>null()
    state%vector_fields=>null()
    state%tensor_fields=>null()
    state%scalar_fields=>null()
    state%meshes=>null()
    state%halos=>null()
    state%csr_sparsities=>null()
    state%csr_matrices=>null()
    state%block_csr_matrices=>null()
    state%option_path=empty_path

  end subroutine nullify_state

  subroutine set_option_path(state, path)
    !!< Set the option path in state.
    type(state_type), intent(inout) :: state
    character(len=*), intent(in) :: path

    state%option_path = trim(path)

  end subroutine set_option_path

  subroutine insert_tensor_field(state, field, name)
    !!< Insert a tensor field into state.
    !!<
    !!< If a field with this name is already present then it is replaced.
    type(state_type), intent(inout) :: state
    type(tensor_field), intent(in) :: field
    character(len=*), intent(in) :: name

    type(tensor_field_pointer), dimension(:), pointer :: tmp_fields
    character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names

    integer :: i
    integer :: old_size

    if (.not.associated(state%tensor_fields)) then
       ! Special case first entry.
       allocate(state%tensor_fields(1))
       allocate(state%tensor_fields(1)%ptr)
       allocate(state%tensor_names(1))

       state%tensor_fields(1)%ptr = field
       state%tensor_names(1) = name
       call incref(field)

    else

       ! Check if the name is already present.
       do i=1,size(state%tensor_fields)
          if (trim(name)==trim(state%tensor_names(i))) then
             ! The name is present!
             call incref(field)
             call deallocate(state%tensor_fields(i)%ptr)
             state%tensor_fields(i)%ptr = field
             return
          end if
       end do

       ! If we get to here then this is a new field.
       tmp_fields=>state%tensor_fields
       tmp_names=>state%tensor_names

       old_size=size(tmp_fields)

       allocate(state%tensor_fields(old_size+1))
       allocate(state%tensor_fields(old_size+1)%ptr)
       allocate(state%tensor_names(old_size+1))

       forall (i=1:old_size)
          state%tensor_fields(i)%ptr => tmp_fields(i)%ptr
       end forall
       state%tensor_names(1:old_size) = tmp_names

       state%tensor_fields(old_size+1)%ptr = field
       state%tensor_names(old_size+1) = name
       call incref(field)

       deallocate(tmp_fields)
       deallocate(tmp_names)

    end if

  end subroutine insert_tensor_field

  subroutine insert_and_alias_tensor_field(state, field, name)
    !!< Insert a tensor field into state(1) and alias it in all others.
    !!<
    !!< If a field with this name is already present then it is replaced.
    type(state_type), dimension(:), intent(inout) :: state
    type(tensor_field), intent(in) :: field
    character(len=*), intent(in) :: name

    type(tensor_field) :: p_field
    integer :: i

    call insert(state(1), field, trim(name))

    p_field=extract_tensor_field(state(1), trim(name))
    p_field%aliased = .true.
    do i = 2, size(state)
      call insert(state(i), p_field, trim(name))
    end do

  end subroutine insert_and_alias_tensor_field

  subroutine insert_vector_field(state, field, name)
    !!< Insert a vector field into state.
    !!<
    !!< If a field with this name is already present then it is replaced.
    type(state_type), intent(inout) :: state
    type(vector_field), intent(in) :: field
    character(len=*), intent(in) :: name

    type(vector_field_pointer), dimension(:), pointer :: tmp_fields
    character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names

    integer :: i
    integer :: old_size

    if (.not.associated(state%vector_fields)) then
       ! Special case first entry.
       allocate(state%vector_fields(1))
       allocate(state%vector_fields(1)%ptr)
       allocate(state%vector_names(1))

       state%vector_fields(1)%ptr = field
       state%vector_names(1) = name

       call incref(field)

    else

       ! Check if the name is already present.
       do i=1,size(state%vector_fields)
          if (trim(name)==trim(state%vector_names(i))) then
             ! The name is present!
             call incref(field)
             call deallocate(state%vector_fields(i)%ptr)
             state%vector_fields(i)%ptr = field
             return
          end if
       end do

       ! If we get to here then this is a new field.
       tmp_fields=>state%vector_fields
       tmp_names=>state%vector_names

       old_size=size(tmp_fields)

       allocate(state%vector_fields(old_size+1))
       allocate(state%vector_fields(old_size+1)%ptr)
       allocate(state%vector_names(old_size+1))

       forall(i=1:old_size)
          state%vector_fields(i)%ptr => tmp_fields(i)%ptr
       end forall
       state%vector_names(1:old_size)= tmp_names

       state%vector_fields(old_size+1)%ptr = field
       state%vector_names(old_size+1) = name
       call incref(field)

       deallocate(tmp_fields)
       deallocate(tmp_names)

    end if

  end subroutine insert_vector_field

  subroutine insert_and_alias_vector_field(state, field, name)
    !!< Insert a vector field into state(1) and alias it in all others.
    !!<
    !!< If a field with this name is already present then it is replaced.
    type(state_type), dimension(:), intent(inout) :: state
    type(vector_field), intent(in) :: field
    character(len=*), intent(in) :: name

    type(vector_field) :: p_field
    integer :: i

    call insert(state(1), field, trim(name))

    p_field=extract_vector_field(state(1), trim(name))
    p_field%aliased = .true.
    do i = 2, size(state)
      call insert(state(i), p_field, trim(name))
    end do

  end subroutine insert_and_alias_vector_field

  subroutine insert_scalar_field(state, field, name)
    !!< Insert a scalar field into state.
    type(state_type), intent(inout) :: state
    type(scalar_field), intent(in) :: field
    character(len=*), intent(in) :: name

    type(scalar_field_pointer), dimension(:), pointer :: tmp_fields
    character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names

    integer :: i
    integer :: old_size

    if (.not.associated(state%scalar_fields)) then
       ! Special case first entry.
       allocate(state%scalar_fields(1))
       allocate(state%scalar_fields(1)%ptr)
       allocate(state%scalar_names(1))

       state%scalar_fields(1)%ptr = field
       state%scalar_names(1) = name

       call incref(field)

    else

       ! Check if the name is already present.
       do i=1,size(state%scalar_fields)
          if (trim(name)==trim(state%scalar_names(i))) then
             ! The name is present!
             call incref(field)
             call deallocate(state%scalar_fields(i)%ptr)
             state%scalar_fields(i)%ptr = field
             return
          end if
       end do

       ! If we get to here then this is a new field.
       tmp_fields=>state%scalar_fields
       tmp_names=>state%scalar_names

       old_size=size(tmp_fields)

       allocate(state%scalar_fields(old_size+1))
       allocate(state%scalar_fields(old_size+1)%ptr)
       allocate(state%scalar_names(old_size+1))

       forall(i=1:old_size)
          state%scalar_fields(i)%ptr => tmp_fields(i)%ptr
       end forall
       state%scalar_names(1:old_size)= tmp_names

       state%scalar_fields(old_size+1)%ptr = field
       state%scalar_names(old_size+1) = name

       call incref(field)

       deallocate(tmp_fields)
       deallocate(tmp_names)

    end if

  end subroutine insert_scalar_field

  subroutine insert_and_alias_scalar_field(state, field, name)
    !!< Insert a scalar field into state(1) and alias it in all others.
    !!<
    !!< If a field with this name is already present then it is replaced.
    type(state_type), dimension(:), intent(inout) :: state
    type(scalar_field), intent(in) :: field
    character(len=*), intent(in) :: name

    type(scalar_field) :: p_field
    integer :: i

    call insert(state(1), field, trim(name))

    p_field=extract_scalar_field(state(1), trim(name))
    p_field%aliased = .true.
    do i = 2, size(state)
      call insert(state(i), p_field, trim(name))
    end do

  end subroutine insert_and_alias_scalar_field

  subroutine insert_mesh(state, mesh, name)
    !!< Insert a mesh into state.
    type(state_type), intent(inout) :: state
    type(mesh_type), intent(in) :: mesh
    character(len=*), intent(in) :: name

    type(mesh_pointer), dimension(:), pointer :: tmp_meshes
    character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names

    integer :: i
    integer :: old_size

    if (.not.associated(state%meshes)) then
       ! Special case first entry.
       allocate(state%meshes(1))
       allocate(state%meshes(1)%ptr)
       allocate(state%mesh_names(1))

       state%meshes(1)%ptr = mesh
       state%mesh_names(1) = name

    else

       ! Check if the name is already present.
       do i=1,size(state%meshes)
          if (trim(name)==trim(state%mesh_names(i))) then
             ! The name is present!
             call deallocate(state%meshes(i)%ptr)
             state%meshes(i)%ptr = mesh
             call incref(mesh)
             return
          end if
       end do

       ! If we get to here then this is a new mesh.
       tmp_meshes=>state%meshes
       tmp_names=>state%mesh_names

       old_size=size(tmp_meshes)

       allocate(state%meshes(old_size+1))
       allocate(state%meshes(old_size+1)%ptr)
       allocate(state%mesh_names(old_size+1))

       forall(i=1:old_size)
          state%meshes(i)%ptr => tmp_meshes(i)%ptr
       end forall
       state%mesh_names(1:old_size)= tmp_names

       state%meshes(old_size+1)%ptr = mesh
       state%mesh_names(old_size+1) = name

       deallocate(tmp_names)
       deallocate(tmp_meshes)

    end if

    call incref(mesh)

  end subroutine insert_mesh

  subroutine insert_and_alias_mesh(state, mesh, name)
    !!< Insert a mesh into state(1) and alias it in all others.
    !!<
    !!< If a field with this name is already present then it is replaced.
    type(state_type), dimension(:), intent(inout) :: state
    type(mesh_type), intent(in) :: mesh
    character(len=*), intent(in) :: name

    type(mesh_type) :: p_mesh
    integer :: i

    call insert(state(1), mesh, trim(name))

    p_mesh=extract_mesh(state(1), trim(name))
    do i = 2, size(state)
      call insert(state(i), p_mesh, trim(name))
    end do

  end subroutine insert_and_alias_mesh

  subroutine insert_halo(state, halo, name)
    !!< Insert a halo into state.
    type(state_type), intent(inout) :: state
    type(halo_type), intent(in) :: halo
    character(len=*), intent(in) :: name

    type(halo_pointer), dimension(:), pointer :: tmp_halos
    character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names

    integer :: i
    integer :: old_size

    if (.not.associated(state%halos)) then
       ! Special case first entry.
       allocate(state%halos(1))
       allocate(state%halos(1)%ptr)
       allocate(state%halo_names(1))

       state%halos(1)%ptr = halo
       state%halo_names(1) = name

    else

       ! Check if the name is already present.
       do i=1,size(state%halos)
          if (trim(name)==trim(state%halo_names(i))) then
             ! The name is present!
             call deallocate(state%halos(i)%ptr)
             state%halos(i)%ptr = halo
             call incref(halo)
             return
          end if
       end do

       ! If we get to here then this is a new halo.
       tmp_halos=>state%halos
       tmp_names=>state%halo_names

       old_size=size(tmp_halos)

       allocate(state%halos(old_size+1))
       allocate(state%halos(old_size+1)%ptr)
       allocate(state%halo_names(old_size+1))

       forall(i=1:old_size)
          state%halos(i)%ptr => tmp_halos(i)%ptr
       end forall
       state%halo_names(1:old_size)= tmp_names

       state%halos(old_size+1)%ptr = halo
       state%halo_names(old_size+1) = name

       deallocate(tmp_names)
       deallocate(tmp_halos)

    end if

    call incref(halo)

  end subroutine insert_halo

  subroutine insert_and_alias_halo(state, halo, name)
    !!< Insert a halo into state(1) and alias it in all others.
    !!<
    !!< If a halo with this name is already present then it is replaced.
    type(state_type), dimension(:), intent(inout) :: state
    type(halo_type), intent(in) :: halo
    character(len=*), intent(in) :: name

    type(halo_type) :: p_halo
    integer :: i

    call insert(state(1), halo, trim(name))

    p_halo=extract_halo(state(1), trim(name))
    do i = 2, size(state)
      call insert(state(i), p_halo, trim(name))
    end do

  end subroutine insert_and_alias_halo

  subroutine insert_csr_sparsity(state, sparsity, name)
    !!< Insert a sparsity into state.
    type(state_type), intent(inout) :: state
    type(csr_sparsity), intent(in) :: sparsity
    character(len=*), intent(in) :: name

    type(csr_sparsity_pointer), dimension(:), pointer :: tmp_csr_sparsities
    character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names

    integer :: i
    integer :: old_size

    if (.not.associated(state%csr_sparsities)) then
       ! Special case first entry.
       allocate(state%csr_sparsities(1))
       allocate(state%csr_sparsities(1)%ptr)
       allocate(state%csr_sparsity_names(1))

       state%csr_sparsities(1)%ptr = sparsity
       state%csr_sparsity_names(1) = name

    else

       ! Check if the name is already present.
       do i=1,size(state%csr_sparsities)
          if (trim(name)==trim(state%csr_sparsity_names(i))) then
             ! The name is present!
             call deallocate(state%csr_sparsities(i)%ptr)
             state%csr_sparsities(i)%ptr = sparsity
             call incref(sparsity)
             return
          end if
       end do

       ! If we get to here then this is a new sparsity.
       tmp_csr_sparsities=>state%csr_sparsities
       tmp_names=>state%csr_sparsity_names

       old_size=size(tmp_csr_sparsities)

       allocate(state%csr_sparsities(old_size+1))
       allocate(state%csr_sparsities(old_size+1)%ptr)
       allocate(state%csr_sparsity_names(old_size+1))

       forall(i=1:old_size)
          state%csr_sparsities(i)%ptr => tmp_csr_sparsities(i)%ptr
       end forall
       state%csr_sparsity_names(1:old_size)= tmp_names

       state%csr_sparsities(old_size+1)%ptr = sparsity
       state%csr_sparsity_names(old_size+1) = name

      deallocate(tmp_names)
      deallocate(tmp_csr_sparsities)

    end if

    call incref(sparsity)

  end subroutine insert_csr_sparsity

  subroutine insert_csr_matrix(state, matrix, name)
    !!< Insert a matrix into state.
    type(state_type), intent(inout) :: state
    type(csr_matrix), intent(in) :: matrix
    character(len=*), intent(in) :: name

    type(csr_matrix_pointer), dimension(:), pointer :: tmp_csr_matrices
    character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names

    integer :: i
    integer :: old_size

    if (.not.associated(state%csr_matrices)) then
       ! Special case first entry.
       allocate(state%csr_matrices(1))
       allocate(state%csr_matrices(1)%ptr)
       allocate(state%csr_matrix_names(1))

       state%csr_matrices(1)%ptr = matrix
       state%csr_matrix_names(1) = name

    else

       ! Check if the name is already present.
       do i=1,size(state%csr_matrices)
          if (trim(name)==trim(state%csr_matrix_names(i))) then
             ! The name is present!
             call deallocate(state%csr_matrices(i)%ptr)
             state%csr_matrices(i)%ptr = matrix
             call incref(matrix)
             return
          end if
       end do

       ! If we get to here then this is a new matrix.
       tmp_csr_matrices=>state%csr_matrices
       tmp_names=>state%csr_matrix_names

       old_size=size(tmp_csr_matrices)

       allocate(state%csr_matrices(old_size+1))
       allocate(state%csr_matrices(old_size+1)%ptr)
       allocate(state%csr_matrix_names(old_size+1))

       forall(i=1:old_size)
          state%csr_matrices(i)%ptr => tmp_csr_matrices(i)%ptr
       end forall
       state%csr_matrix_names(1:old_size)= tmp_names

       state%csr_matrices(old_size+1)%ptr = matrix
       state%csr_matrix_names(old_size+1) = name

       deallocate(tmp_names)
       deallocate(tmp_csr_matrices)

    end if

    call incref(matrix)

  end subroutine insert_csr_matrix

  subroutine insert_and_alias_csr_matrix(state, matrix, name)
    !!< Insert a matrix into all states
    type(state_type), dimension(:), intent(inout) :: state
    type(csr_matrix), intent(in) :: matrix
    character(len=*), intent(in) :: name

    type(csr_matrix) :: p_matrix
    integer :: i

    ! insert into state(1)
    call insert(state(1), matrix, trim(name))

    p_matrix=extract_csr_matrix(state(1), trim(name))

    do i = 2, size(state)
      call insert(state(i), p_matrix, trim(name))
    end do

  end subroutine insert_and_alias_csr_matrix

  subroutine insert_and_alias_csr_sparsity(state, sparsity, name)
    !!< Insert a sparsity into state all states
    type(state_type), dimension(:), intent(inout) :: state
    type(csr_sparsity), intent(in) :: sparsity
    character(len=*), intent(in) :: name

    type(csr_sparsity) :: p_sparsity
    integer :: i

    ! insert into state(1)
    call insert(state(1), sparsity, trim(name))

    p_sparsity=extract_csr_sparsity(state(1), trim(name))

    do i = 2, size(state)
       call insert(state(i), p_sparsity, trim(name))
    end do

  end subroutine insert_and_alias_csr_sparsity

  subroutine insert_block_csr_matrix(state, matrix, name)
    !!< Insert a block matrix into state.
    type(state_type), intent(inout) :: state
    type(block_csr_matrix), intent(in) :: matrix
    character(len=*), intent(in) :: name

    type(block_csr_matrix_pointer), dimension(:), pointer :: tmp_block_csr_matrices
    character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names

    integer :: i
    integer :: old_size

    if (.not.associated(state%block_csr_matrices)) then
       ! Special case first entry.
       allocate(state%block_csr_matrices(1))
       allocate(state%block_csr_matrices(1)%ptr)
       allocate(state%block_csr_matrix_names(1))

       state%block_csr_matrices(1)%ptr = matrix
       state%block_csr_matrix_names(1) = name

    else

       ! Check if the name is already present.
       do i=1,size(state%block_csr_matrices)
          if (trim(name)==trim(state%block_csr_matrix_names(i))) then
             ! The name is present!
             call deallocate(state%block_csr_matrices(i)%ptr)
             state%block_csr_matrices(i)%ptr = matrix
             call incref(matrix)
             return
          end if
       end do

       ! If we get to here then this is a new matrix.
       tmp_block_csr_matrices=>state%block_csr_matrices
       tmp_names=>state%block_csr_matrix_names

       old_size=size(tmp_block_csr_matrices)

       allocate(state%block_csr_matrices(old_size+1))
       allocate(state%block_csr_matrices(old_size+1)%ptr)
       allocate(state%block_csr_matrix_names(old_size+1))

       forall(i=1:old_size)
          state%block_csr_matrices(i)%ptr => tmp_block_csr_matrices(i)%ptr
       end forall
       state%block_csr_matrix_names(1:old_size)= tmp_names

       state%block_csr_matrices(old_size+1)%ptr = matrix
       state%block_csr_matrix_names(old_size+1) = name

       deallocate(tmp_names)
       deallocate(tmp_block_csr_matrices)

    end if

    call incref(matrix)

  end subroutine insert_block_csr_matrix

  subroutine insert_and_alias_block_csr_matrix(state, matrix, name)
    !!< Insert a matrix into state.
    type(state_type), dimension(:), intent(inout) :: state
    type(block_csr_matrix), intent(in) :: matrix
    character(len=*), intent(in) :: name

    type(block_csr_matrix) :: p_matrix
    integer :: i

    ! insert into state(1)
    call insert(state(1), matrix, trim(name))

    p_matrix=extract_block_csr_matrix(state(1), trim(name))

    do i = 2, size(state)
      call insert(state(i), p_matrix, trim(name))
    end do

  end subroutine insert_and_alias_block_csr_matrix

  subroutine remove_tensor_field(state, name, stat)
  type(state_type), intent(inout) :: state
  character(len=*), intent(in) :: name
  integer, optional, intent(out) :: stat

  type(tensor_field_pointer), dimension(:), pointer :: tmp_fields
  character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names
  integer :: idx, i, j, old_size

  if (.not. has_tensor_field(state, name)) then
    if (present(stat)) then
      stat=1
    else
       ewrite(-1,*) "State: "//trim(state%name)
       ewrite(-1,*) "Field name: "//trim(name)
      FLExit("You're trying to remove a tensor field .. that isn't there!")
    end if
  end if

  tmp_fields=>state%tensor_fields
  tmp_names=>state%tensor_names

  old_size=size(tmp_fields)

  do i=1,old_size
    if (trim(tmp_names(i)) == name) then
      idx = i
      exit
    end if
  end do

  allocate(state%tensor_fields(old_size-1))
  allocate(state%tensor_names(old_size-1))

  j = 0
  do i=1,old_size
    if (i /= idx) then
      j = j + 1
      state%tensor_fields(j)%ptr => tmp_fields(i)%ptr
      state%tensor_names(j) = tmp_names(i)
    end if
  end do

  call deallocate(tmp_fields(idx)%ptr)
  deallocate(tmp_fields(idx)%ptr)

  deallocate(tmp_fields)
  deallocate(tmp_names)

  end subroutine remove_tensor_field

  subroutine remove_vector_field(state, name, stat)
  type(state_type), intent(inout) :: state
  character(len=*), intent(in) :: name
  integer, optional, intent(out) :: stat

  type(vector_field_pointer), dimension(:), pointer :: tmp_fields
  character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names
  integer :: idx, i, j, old_size

  if (.not. has_vector_field(state, name)) then
    if (present(stat)) then
      stat=1
    else
       ewrite(-1,*) "State: "//trim(state%name)
       ewrite(-1,*) "Field name: "//trim(name)
      FLExit("You're trying to remove a vector field .. that isn't there!")
    end if
  end if

  tmp_fields=>state%vector_fields
  tmp_names=>state%vector_names

  old_size=size(tmp_fields)

  do i=1,old_size
    if (trim(tmp_names(i)) == name) then
      idx = i
      exit
    end if
  end do

  allocate(state%vector_fields(old_size-1))
  allocate(state%vector_names(old_size-1))

  j = 0
  do i=1,old_size
    if (i /= idx) then
      j = j + 1
      state%vector_fields(j)%ptr => tmp_fields(i)%ptr
      state%vector_names(j) = tmp_names(i)
    end if
  end do

  call deallocate(tmp_fields(idx)%ptr)
  deallocate(tmp_fields(idx)%ptr)

  deallocate(tmp_fields)
  deallocate(tmp_names)

  end subroutine remove_vector_field

  subroutine remove_scalar_field(state, name, stat)
  type(state_type), intent(inout) :: state
  character(len=*), intent(in) :: name
  integer, optional, intent(out) :: stat

  type(scalar_field_pointer), dimension(:), pointer :: tmp_fields
  character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names
  integer :: idx, i, j, old_size

  if (.not. has_scalar_field(state, name)) then
    if (present(stat)) then
      stat=1
    else
       ewrite(-1,*) "State: "//trim(state%name)
       ewrite(-1,*) "Field name: "//trim(name)
      FLExit("You're trying to remove a scalar field .. that isn't there!")
    end if
  end if

  tmp_fields=>state%scalar_fields
  tmp_names=>state%scalar_names

  old_size=size(tmp_fields)

  do i=1,old_size
    if (trim(tmp_names(i)) == name) then
      idx = i
      exit
    end if
  end do

  allocate(state%scalar_fields(old_size-1))
  allocate(state%scalar_names(old_size-1))

  j = 0
  do i=1,old_size
    if (i /= idx) then
      j = j + 1
      state%scalar_fields(j)%ptr => tmp_fields(i)%ptr
      state%scalar_names(j) = tmp_names(i)
    end if
  end do

  call deallocate(tmp_fields(idx)%ptr)
  deallocate(tmp_fields(idx)%ptr)

  deallocate(tmp_fields)
  deallocate(tmp_names)

  end subroutine remove_scalar_field

  subroutine remove_csr_sparsity(state, name, stat)
  type(state_type), intent(inout) :: state
  character(len=*), intent(in) :: name
  integer, optional, intent(out) :: stat

  type(csr_sparsity_pointer), dimension(:), pointer :: tmp_fields
  character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names
  integer :: idx, i, j, old_size

  if (.not. has_csr_sparsity(state, name)) then
    if (present(stat)) then
      stat=1
    else
       ewrite(-1,*) "State: "//trim(state%name)
       ewrite(-1,*) "Sparsity name: "//trim(name)
      FLExit("You're trying to remove a csr sparsity .. that isn't there!")
    end if
  end if

  tmp_fields=>state%csr_sparsities
  tmp_names=>state%csr_sparsity_names

  old_size=size(tmp_fields)

  do i=1,old_size
    if (trim(tmp_names(i)) == name) then
      idx = i
      exit
    end if
  end do

  allocate(state%csr_sparsities(old_size-1))
  allocate(state%csr_sparsity_names(old_size-1))

  j = 0
  do i=1,old_size
    if (i /= idx) then
      j = j + 1
      state%csr_sparsities(j)%ptr => tmp_fields(i)%ptr
      state%csr_sparsity_names(j) = tmp_names(i)
    end if
  end do

  call deallocate(tmp_fields(idx)%ptr)
  deallocate(tmp_fields(idx)%ptr)

  deallocate(tmp_fields)
  deallocate(tmp_names)

  end subroutine remove_csr_sparsity

  subroutine remove_csr_matrix(state, name, stat)
  type(state_type), intent(inout) :: state
  character(len=*), intent(in) :: name
  integer, optional, intent(out) :: stat

  type(csr_matrix_pointer), dimension(:), pointer :: tmp_fields
  character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names
  integer :: idx, i, j, old_size

  if (.not. has_csr_matrix(state, name)) then
    if (present(stat)) then
      stat=1
   else
      ewrite(-1,*) "State: "//trim(state%name)
       ewrite(-1,*) "Matrix name: "//trim(name)
      FLExit("You're trying to remove a csr matrix .. that isn't there!")
    end if
  end if

  tmp_fields=>state%csr_matrices
  tmp_names=>state%csr_matrix_names

  old_size=size(tmp_fields)

  do i=1,old_size
    if (trim(tmp_names(i)) == name) then
      idx = i
      exit
    end if
  end do

  allocate(state%csr_matrices(old_size-1))
  allocate(state%csr_matrix_names(old_size-1))

  j = 0
  do i=1,old_size
    if (i /= idx) then
      j = j + 1
      state%csr_matrices(j)%ptr => tmp_fields(i)%ptr
      state%csr_matrix_names(j) = tmp_names(i)
    end if
  end do

  call deallocate(tmp_fields(idx)%ptr)
  deallocate(tmp_fields(idx)%ptr)

  deallocate(tmp_fields)
  deallocate(tmp_names)

  end subroutine remove_csr_matrix

  subroutine remove_block_csr_matrix(state, name, stat)
  type(state_type), intent(inout) :: state
  character(len=*), intent(in) :: name
  integer, optional, intent(out) :: stat

  type(block_csr_matrix_pointer), dimension(:), pointer :: tmp_fields
  character(len=FIELD_NAME_LEN), dimension(:), pointer :: tmp_names
  integer :: idx, i, j, old_size

  if (.not. has_block_csr_matrix(state, name)) then
    if (present(stat)) then
      stat=1
    else
       ewrite(-1,*) "State: "//trim(state%name)
       ewrite(-1,*) "Matrix name: "//trim(name)
      FLExit("You're trying to remove a block csr matrix .. that isn't there!")
    end if
  end if

  tmp_fields=>state%block_csr_matrices
  tmp_names=>state%block_csr_matrix_names

  old_size=size(tmp_fields)

  do i=1,old_size
    if (trim(tmp_names(i)) == name) then
      idx = i
      exit
    end if
  end do

  allocate(state%block_csr_matrices(old_size-1))
  allocate(state%block_csr_matrix_names(old_size-1))

  j = 0
  do i=1,old_size
    if (i /= idx) then
      j = j + 1
      state%block_csr_matrices(j)%ptr => tmp_fields(i)%ptr
      state%block_csr_matrix_names(j) = tmp_names(i)
    end if
  end do

  call deallocate(tmp_fields(idx)%ptr)
  deallocate(tmp_fields(idx)%ptr)

  deallocate(tmp_fields)
  deallocate(tmp_names)

  end subroutine remove_block_csr_matrix

  function field_rank(state, name, stat)
    !!< Return the rank of the named field in state

    type(state_type), intent(in) :: state
    character(len = *), intent(in) :: name
    integer, optional, intent(out) :: stat

    integer :: field_rank

    logical :: s_field, v_field, t_field

    if(present(stat)) stat = 0

    s_field = has_scalar_field(state, name)
    v_field = has_vector_field(state, name)
    t_field = has_tensor_field(state, name)

    if(count((/s_field, v_field, t_field/)) > 1)  then
      if(present(stat)) then
        stat = 2
      else
        FLAbort("Multiple field types found for field " // trim(name))
      end if
    else if(s_field) then
      field_rank = 0
    else if(v_field) then
      field_rank = 1
    else if(t_field) then
      field_rank = 2
    else
      if(present(stat)) then
        stat = 1
      else
        FLExit(trim(name) // " is not a field name in this state")
      end if
    end if

  end function field_rank

  function extract_tensor_field(state, name, stat) result (field)
    !!< Return a pointer to the tensor field with the correct name.
    type(tensor_field), pointer :: field
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i

    if (present(stat)) stat=0
    field => fake_tensor_field

    if (associated(state%tensor_fields)) then
       do i=1,size(state%tensor_fields)
          if (trim(name)==trim(state%tensor_names(i))) then
             ! Found the right field

             field=>state%tensor_fields(i)%ptr
             return
          end if
       end do
    end if

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
      if (associated(state%tensor_names)) then
        do i=1,size(state%tensor_names)
          ewrite(-1,*) "i: ", i, " -- ", state%tensor_names(i)
        end do
      end if
      FLExit(trim(name)//" is not a field name in this state")
    end if

  end function extract_tensor_field

  function extract_from_one_vector_field(state, name, stat) result (field)
    !!< Return a pointer to the vector field with the correct name.
    type(vector_field), pointer :: field
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i

    if (present(stat)) stat=0
    field => fake_vector_field

    if (associated(state%vector_fields)) then
       do i=1,size(state%vector_fields)
          if (trim(name)==trim(state%vector_names(i))) then
             ! Found the right field

             field=>state%vector_fields(i)%ptr
             return
          end if
       end do
    end if

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
      if (associated(state%vector_names)) then
        do i=1,size(state%vector_names)
          ewrite(-1,*) "i: ", i, " -- ", state%vector_names(i)
        end do
      end if
      FLExit(trim(name)//" is not a field name in this state")
    end if

  end function extract_from_one_vector_field

  function extract_from_one_scalar_field(state, name, stat, allocated) result (field)
    !!< Return a pointer to the scalar field with the correct name.
    type(scalar_field), pointer :: field
    type(scalar_field), pointer :: mem
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat
    logical, intent(out), optional :: allocated
    type(vector_field), pointer :: vfield

    integer :: i, lstat, idx, dim

    if (present(stat)) stat=0
    field => fake_scalar_field

    ! if the allocated flag is present
    ! then you can look in the vector and tensor
    ! fields for names of the form
    ! Velocity%1
    if (present(allocated)) then
      allocated = .false.
      idx = index(name, "%")
      if (idx /= 0) then
        read(name(idx+1:len(name)), *) dim
        vfield => extract_vector_field(state, name(1:idx-1), lstat)
        if (lstat == 0) then
          allocate(mem)
          allocated = .true.
          mem = extract_scalar_field(vfield, dim, lstat)
          if (lstat /= 0) then
            deallocate(mem)
            allocated = .false.
            if (present(stat)) then
              stat = 1
            else
              ewrite(-1,*) "name: ", name
              FLExit("Couldn't find vector/tensor component!")
            end if
          end if
          field => mem
          return
        end if
      end if
    end if

    if (associated(state%scalar_fields)) then
       do i=1,size(state%scalar_fields)
          if (trim(name)==trim(state%scalar_names(i))) then
             ! Found the right field

             field=>state%scalar_fields(i)%ptr
             return
          end if
       end do
    end if

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
      if (associated(state%scalar_names)) then
        do i=1,size(state%scalar_names)
          ewrite(-1,*) "i: ", i, " -- ", state%scalar_names(i)
        end do
      end if
      FLExit(trim(name)//" is not a field name in this state")
    end if

  end function extract_from_one_scalar_field

  function extract_from_any_scalar_field(state, name, stat, allocated) result (field)
    !!< Return a pointer to the scalar field with the correct name.
    type(scalar_field), pointer :: field
    type(scalar_field), pointer :: mem
    type(state_type), dimension(:), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat
    logical, intent(out), optional :: allocated
    type(vector_field), pointer :: vfield

    integer :: i, j, lstat, idx, dim

    if (present(stat)) stat=0
    field => fake_scalar_field

    ! if the allocated flag is present
    ! then you can look in the vector and tensor
    ! fields for names of the form
    ! Velocity%1
    if (present(allocated)) then
      allocated = .false.
      idx = index(name, "%")
      if (idx /= 0) then
        read(name(idx+1:len(name)), *) dim
        do i = 1, size(state)
          vfield => extract_vector_field(state(i), name(1:idx-1), lstat)
          if (lstat == 0) then
            allocate(mem)
            allocated = .true.
            mem = extract_scalar_field(vfield, dim, lstat)
            if (lstat /= 0) then
              deallocate(mem)
              allocated = .false.
              if (present(stat)) then
                stat = 1
              else
                ewrite(-1,*) "name: ", name
                FLExit("Couldn't find vector/tensor component!")
              end if
            end if
            field => mem
            return
          end if
        end do
      end if
    end if

    do i = 1, size(state)
      if (associated(state(i)%scalar_fields)) then
        do j=1,size(state(i)%scalar_fields)
            if (trim(name)==trim(state(i)%scalar_names(j))) then
              ! Found the right field

              field=>state(i)%scalar_fields(j)%ptr
              return
            end if
        end do
      end if
    end do

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
      do i = 1, size(state)
        if (associated(state(i)%scalar_names)) then
          do j=1,size(state(i)%scalar_names)
            ewrite(-1,*) "i, j: ", i, j, " -- ", state(i)%scalar_names(j)
          end do
        end if
      end do
      FLExit(trim(name)//" is not a field name in these states")
    end if

  end function extract_from_any_scalar_field

  function extract_from_any_vector_field(state, name, stat) result (field)
    !!< Return a pointer to the vector field with the correct name.
    type(vector_field), pointer :: field
    type(state_type), dimension(:), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i, j

    if (present(stat)) stat=0
    field => fake_vector_field

    do i = 1, size(state)
      if (associated(state(i)%vector_fields)) then
        do j=1,size(state(i)%vector_fields)
            if (trim(name)==trim(state(i)%vector_names(j))) then
              ! Found the right field

              field=>state(i)%vector_fields(j)%ptr
              return
            end if
        end do
      end if
    end do

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
      do i = 1, size(state)
        if (associated(state(i)%vector_names)) then
          do j=1,size(state(i)%vector_names)
            ewrite(-1,*) "i, j: ", i, j, " -- ", state(i)%vector_names(j)
          end do
        end if
      end do
      FLExit(trim(name)//" is not a field name in these states")
    end if

  end function extract_from_any_vector_field

  function extract_field_mesh(state, name, stat) result(mesh)
    !!< Return the mesh for the named field in state

    type(state_type), intent(in) :: state
    character(len = *), intent(in) :: name
    integer, optional, intent(out) :: stat

    type(mesh_type), pointer :: mesh

    integer :: s_stat, v_stat, t_stat
    type(scalar_field), pointer :: s_field
    type(tensor_field), pointer :: t_field
    type(vector_field), pointer :: v_field

    if(present(stat)) stat = 0

    s_field => extract_scalar_field(state, name, s_stat)
    v_field => extract_vector_field(state, name, v_stat)
    t_field => extract_tensor_field(state, name, t_stat)

    if(count((/s_stat == 0, v_stat == 0, t_stat == 0/)) > 1) then
      if(present(stat)) then
        stat = 2
      else
        FLAbort("Multiple field types found for field " // trim(name))
      end if
    else if(s_stat == 0) then
      mesh => s_field%mesh
    else if(v_stat == 0) then
      mesh => v_field%mesh
    else if(t_stat == 0) then
      mesh => t_field%mesh
    else
      if(present(stat)) then
        stat = 1
      else
        FLExit(trim(name) // " is not a field name in this state")
      end if
    end if

  end function extract_field_mesh

  function extract_mesh_from_one(state, name, stat) result (mesh)
    !!< Return a pointer to the mesh with the correct name.
    type(mesh_type), pointer :: mesh
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i

    if (present(stat)) stat=0
    mesh => fake_mesh

    if (associated(state%meshes)) then
       do i=1,size(state%meshes)
          if (trim(name)==trim(state%mesh_names(i))) then
             ! Found the right field

             mesh=>state%meshes(i)%ptr
             return
          end if
       end do
    end if

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
       FLExit(trim(name)//" is not a mesh name in this state")
    end if

  end function extract_mesh_from_one

  function extract_mesh_from_any(state, name, stat) result (mesh)
    !!< Return a pointer to the mesh with the correct name.
    type(mesh_type), pointer :: mesh
    type(state_type), dimension(:), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i, j

    if (present(stat)) stat=0
    mesh => fake_mesh

    do i = 1, size(state)
      if (associated(state(i)%meshes)) then
        do j=1,size(state(i)%meshes)
            if (trim(name)==trim(state(i)%mesh_names(j))) then
              ! Found the right field

              mesh=>state(i)%meshes(j)%ptr
              return
            end if
        end do
      end if
    end do

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
       FLExit(trim(name)//" is not a mesh name in these state")
    end if

  end function extract_mesh_from_any

  function extract_halo(state, name, stat) result (halo)
    !!< Return a pointer to the halo with the correct name.
    type(halo_type), pointer :: halo
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i

    if (present(stat)) stat=0
    halo => fake_halo

    if (associated(state%halos)) then
       do i=1,size(state%halos)
          if (trim(name)==trim(state%halo_names(i))) then
             ! Found the right halo

             halo=>state%halos(i)%ptr
             return
          end if
       end do
    end if

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
       FLExit(trim(name)//" is not a halo name in this state")
    end if

  end function extract_halo

  function extract_from_one_csr_sparsity(state, name, stat) result (sparsity)
    !!< Return a pointer to the sparsity with the correct name.
    type(csr_sparsity), pointer :: sparsity
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i

    if (present(stat)) stat=0
    sparsity => fake_csr_sparsity

    if (associated(state%csr_sparsities)) then
       do i=1,size(state%csr_sparsities)
          if (trim(name)==trim(state%csr_sparsity_names(i))) then
             ! Found the right field

             sparsity=>state%csr_sparsities(i)%ptr
             return
          end if
       end do
    end if

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
       FLExit(trim(name)//" is not a sparsity name in this state")
    end if

  end function extract_from_one_csr_sparsity

  function extract_from_any_csr_sparsity(state, name, stat) result (sparsity)
    !!< Return a pointer to the sparsity with the correct name.
    type(csr_sparsity), pointer :: sparsity
    type(state_type), dimension(:), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i, j

    if (present(stat)) stat=0
    sparsity => fake_csr_sparsity

    do i = 1, size(state)
      if (associated(state(i)%csr_sparsities)) then
        do j=1,size(state(i)%csr_sparsities)
            if (trim(name)==trim(state(i)%csr_sparsity_names(j))) then
              ! Found the right field

              sparsity=>state(i)%csr_sparsities(j)%ptr
              return
            end if
        end do
      end if
    end do

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
       FLExit(trim(name)//" is not a sparsity name in this state")
    end if

  end function extract_from_any_csr_sparsity

  function extract_from_one_csr_matrix(state, name, stat) result (matrix)
    !!< Return a pointer to the matrix with the correct name.
    type(csr_matrix), pointer :: matrix
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i

    if (present(stat)) stat=0
    matrix => fake_csr_matrix

    if (associated(state%csr_matrices)) then
       do i=1,size(state%csr_matrices)
          if (trim(name)==trim(state%csr_matrix_names(i))) then
             ! Found the right field
             matrix=>state%csr_matrices(i)%ptr
             return
          end if
       end do
    end if

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
       FLExit(trim(name)//" is not a matrix name in this state")
    end if

  end function extract_from_one_csr_matrix

  function extract_from_any_csr_matrix(state, name, stat) result (matrix)
    !!< Return a pointer to the matrix with the correct name.
    type(csr_matrix), pointer :: matrix
    type(state_type), dimension(:), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i, j

    if (present(stat)) stat=0
    matrix => fake_csr_matrix

    do i = 1, size(state)
      if (associated(state(i)%csr_matrices)) then
         do j=1,size(state(i)%csr_matrices)
            if (trim(name)==trim(state(i)%csr_matrix_names(j))) then
               ! Found the right field
               matrix=>state(i)%csr_matrices(j)%ptr
               return
            end if
         end do
      end if
    end do

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
       FLExit(trim(name)//" is not a matrix name in these states")
    end if

  end function extract_from_any_csr_matrix

  function extract_from_one_block_csr_matrix(state, name, stat) result (matrix)
    !!< Return a pointer to the block matrix with the correct name.
    type(block_csr_matrix), pointer :: matrix
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i

    if (present(stat)) stat=0
    matrix => fake_block_csr_matrix

    if (associated(state%block_csr_matrices)) then
       do i=1,size(state%block_csr_matrices)
          if (trim(name)==trim(state%block_csr_matrix_names(i))) then
             ! Found the right field
             matrix=>state%block_csr_matrices(i)%ptr
             return
          end if
       end do
    end if

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
       FLExit(trim(name)//" is not a block matrix name in this state")
    end if

  end function extract_from_one_block_csr_matrix

  function extract_from_any_block_csr_matrix(state, name, stat) result (matrix)
    !!< Return a pointer to the block matrix with the correct name.
    type(block_csr_matrix), pointer :: matrix
    type(state_type), dimension(:), intent(in) :: state
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    integer :: i, j

    if (present(stat)) stat=0
    matrix => fake_block_csr_matrix

    do i = 1, size(state)
      if (associated(state(i)%block_csr_matrices)) then
         do j=1,size(state(i)%block_csr_matrices)
            if (trim(name)==trim(state(i)%block_csr_matrix_names(j))) then
               ! Found the right field
               matrix=>state(i)%block_csr_matrices(j)%ptr
               return
            end if
         end do
      end if
    end do

    ! We didn't find name!
    if (present(stat)) then
       stat=1
    else
       FLExit(trim(name)//" is not a block matrix name in these states")
    end if

  end function extract_from_any_block_csr_matrix

  function extract_scalar_field_by_index(state, index) result (field)
    !!< Return a pointer to the indexth scalar field in state.
    !!< This is primarily useful for looping over all the fields in a
    !!< state.
    type(scalar_field), pointer  :: field
    type(state_type), intent(in) :: state
    integer, intent(in) :: index

    assert(index<=scalar_field_count(state))

    field=>state%scalar_fields(index)%ptr

  end function extract_scalar_field_by_index

  function extract_vector_field_by_index(state, index) result (field)
    !!< Return a pointer to the indexth vector field in state.
    !!< This is primarily useful for looping over all the fields in a
    !!< state.
    type(vector_field), pointer  :: field
    type(state_type), intent(in) :: state
    integer, intent(in) :: index

    assert(index<=vector_field_count(state))

    field=>state%vector_fields(index)%ptr

  end function extract_vector_field_by_index

  function extract_tensor_field_by_index(state, index) result (field)
    !!< Return a pointer to the indexth tensor field in state.
    !!< This is primarily useful for looping over all the fields in a
    !!< state.
    type(tensor_field), pointer  :: field
    type(state_type), intent(in) :: state
    integer, intent(in) :: index

    assert(index<=tensor_field_count(state))

    field=>state%tensor_fields(index)%ptr

  end function extract_tensor_field_by_index

  function extract_mesh_by_index(state, index) result (field)
    !!< Return a pointer to the indexth mesh in state.
    !!< This is primarily useful for looping over all the fields in a
    !!< state.
    type(mesh_type), pointer  :: field
    type(state_type), intent(in) :: state
    integer, intent(in) :: index

    assert(index<=mesh_count(state))

    field=>state%meshes(index)%ptr

  end function extract_mesh_by_index

  function extract_halo_by_index(state, index) result (field)
    !!< Return a pointer to the indexth halo in state.
    !!< This is primarily useful for looping over all the halos in a
    !!< state.
    type(halo_type), pointer  :: field
    type(state_type), intent(in) :: state
    integer, intent(in) :: index

    assert(index<=halo_count(state))

    field=>state%halos(index)%ptr

  end function extract_halo_by_index

  function extract_csr_sparsity_by_index(state, index) result (field)
    !!< Return a pointer to the indexth csr sparsity in state.
    !!< This is primarily useful for looping over all the fields in a
    !!< state.
    type(csr_sparsity), pointer  :: field
    type(state_type), intent(in) :: state
    integer, intent(in) :: index

    assert(index<=csr_sparsity_count(state))

    field=>state%csr_sparsities(index)%ptr

  end function extract_csr_sparsity_by_index

  function extract_csr_matrix_by_index(state, index) result (field)
    !!< Return a pointer to the indexth csr matrix in state.
    !!< This is primarily useful for looping over all the fields in a
    !!< state.
    type(csr_matrix), pointer  :: field
    type(state_type), intent(in) :: state
    integer, intent(in) :: index

    assert(index<=csr_matrix_count(state))

    field=>state%csr_matrices(index)%ptr

  end function extract_csr_matrix_by_index

  function extract_block_csr_matrix_by_index(state, index) result (field)
    !!< Return a pointer to the indexth block csr matrix in state.
    !!< This is primarily useful for looping over all the fields in a
    !!< state.
    type(block_csr_matrix), pointer  :: field
    type(state_type), intent(in) :: state
    integer, intent(in) :: index

    assert(index<=block_csr_matrix_count(state))

    field=>state%block_csr_matrices(index)%ptr

  end function extract_block_csr_matrix_by_index

  function has_scalar_field(state, name) result(present)
    !!< Return true if there is a field named name in state.
    logical :: present
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name

    if (associated(state%scalar_names)) then
      present=any(trim(name)==state%scalar_names)
    else
      present=.false.
    end if

  end function has_scalar_field

  function has_vector_field(state, name) result(present)
    !!< Return true if there is a field named name in state.
    logical :: present
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name

    if (associated(state%vector_names)) then
      present=any(trim(name)==state%vector_names)
    else
      present=.false.
    end if

  end function has_vector_field

  function has_tensor_field(state, name) result(present)
    !!< Return true if there is a field named name in state.
    logical :: present
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name

    if (associated(state%tensor_names)) then
      present=any(trim(name)==state%tensor_names)
    else
      present=.false.
    end if

  end function has_tensor_field

  function has_mesh(state, name) result(present)
    !!< Return true if there is a mesh named name in state.
    logical :: present
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name

    if (associated(state%mesh_names)) then
      present=any(trim(name)==state%mesh_names)
    else
      present=.false.
    end if

  end function has_mesh

  function state_has_halo(state, name) result(present)
    !!< Return true if there is a halo named name in state.
    logical :: present
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name

    if (associated(state%halo_names)) then
      present=any(trim(name)==state%halo_names)
    else
      present=.false.
    end if

  end function state_has_halo

  function has_csr_sparsity(state, name) result(present)
    !!< Return true if there is a sparsity named name in state.
    logical :: present
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name

    if (associated(state%csr_sparsity_names)) then
      present=any(trim(name)==state%csr_sparsity_names)
    else
      present=.false.
    end if

  end function has_csr_sparsity

  function has_csr_matrix(state, name) result(present)
    !!< Return true if there is a matrix named name in state.
    logical :: present
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name

    if (associated(state%csr_matrix_names)) then
      present=any(trim(name)==state%csr_matrix_names)
    else
      present=.false.
    end if

  end function has_csr_matrix

  function has_block_csr_matrix(state, name) result(present)
    !!< Return true if there is a matrix named name in state.
    logical :: present
    type(state_type), intent(in) :: state
    character(len=*), intent(in) :: name

    if (associated(state%block_csr_matrix_names)) then
      present=any(trim(name)==state%block_csr_matrix_names)
    else
      present=.false.
    end if

  end function has_block_csr_matrix

  pure function field_count(state)
    integer field_count
    type(state_type), intent(in) :: state

    field_count = scalar_field_count(state) + &
                  vector_field_count(state) + &
                  tensor_field_count(state)

  end function field_count

  pure function scalar_field_count(state)
    !!< Return the number of scalar fields in state.
    integer :: scalar_field_count
    type(state_type), intent(in) :: state

    if (associated(state%scalar_fields)) then
       scalar_field_count=size(state%scalar_fields)
    else
       scalar_field_count=0
    end if

  end function scalar_field_count

  pure function vector_field_count(state)
    !!< Return the number of vector fields in state.
    integer :: vector_field_count
    type(state_type), intent(in) :: state

    if (associated(state%vector_fields)) then
       vector_field_count=size(state%vector_fields)
    else
       vector_field_count=0
    end if

  end function vector_field_count

  pure function tensor_field_count(state)
    !!< Return the number of tensor fields in state.
    integer :: tensor_field_count
    type(state_type), intent(in) :: state

    if (associated(state%tensor_fields)) then
       tensor_field_count=size(state%tensor_fields)
    else
       tensor_field_count=0
    end if

  end function tensor_field_count

  pure function mesh_count(state)
    !!< Return the number of meshes in state.
    integer :: mesh_count
    type(state_type), intent(in) :: state

    if (associated(state%meshes)) then
       mesh_count=size(state%meshes)
    else
       mesh_count=0
    end if

  end function mesh_count

  pure function halo_count_state(state) result(halo_count)
    !!< Return the number of halos in state.
    integer :: halo_count
    type(state_type), intent(in) :: state

    if (associated(state%halos)) then
       halo_count=size(state%halos)
    else
       halo_count=0
    end if

  end function halo_count_state

  pure function csr_sparsity_count(state)
    !!< Return the number of csr_sparsities in state.
    integer :: csr_sparsity_count
    type(state_type), intent(in) :: state

    if (associated(state%csr_sparsities)) then
       csr_sparsity_count=size(state%csr_sparsities)
    else
       csr_sparsity_count=0
    end if

  end function csr_sparsity_count

  pure function csr_matrix_count(state)
    !!< Return the number of csr_matrices in state.
    integer :: csr_matrix_count
    type(state_type), intent(in) :: state

    if (associated(state%csr_matrices)) then
       csr_matrix_count=size(state%csr_matrices)
    else
       csr_matrix_count=0
    end if

  end function csr_matrix_count

  pure function block_csr_matrix_count(state)
    !!< Return the number of csr_matrices in state.
    integer :: block_csr_matrix_count
    type(state_type), intent(in) :: state

    if (associated(state%block_csr_matrices)) then
       block_csr_matrix_count=size(state%block_csr_matrices)
    else
       block_csr_matrix_count=0
    end if

  end function block_csr_matrix_count

  subroutine set_vector_field_in_state(state, to_field_name, from_field_name)
    !!< Set the value of to_field to the value of from_field.
    type(state_type), intent(inout) :: state
    character(len=*), intent(in) :: to_field_name, from_field_name

    type(vector_field), pointer :: to_field, from_field

    to_field=>extract_vector_field(state, to_field_name)
    from_field=>extract_vector_field(state, from_field_name)

    call set(to_field, from_field)

  end subroutine set_vector_field_in_state

  integer function get_state_index(states, name, stat)
  !!< Auxillary function to search in a set of states by name
    type(state_type), dimension(:), intent(in):: states
    character(len=*), intent(in):: name
    integer, optional, intent(out):: stat

    integer i

    do i=1, size(states)
      if (states(i)%name==name) then

        get_state_index=i
        if (present(stat)) stat=0
        return

      end if
    end do
    !! failed to find
    if (present(stat)) then
      stat=1
    else
      FLExit(name//" is not the name of any of the given states.")
    end if

  end function get_state_index

  subroutine print_state(state, unit)
  !!< Prints the names of all objects in state
    type(state_type), intent(in) :: state
    integer, intent(in), optional :: unit
    integer :: i, lunit

    if (present(unit)) then
       lunit=unit
    else
       lunit=0
    end if

    write(lunit,'(a)') "State: "// trim(state%name)

    write(lunit,'(a)') "Meshes: "
    if (associated(state%mesh_names)) then
      do i=1,size(state%mesh_names)
        write(lunit,'(a)') " +" // trim(state%mesh_names(i))
      end do
    else
      write(lunit, '(a)') " none"
    end if

    write(lunit,'(a)') "Halos: "
    if (associated(state%halo_names)) then
      do i=1,size(state%halo_names)
        write(lunit,'(a)') " +" // trim(state%halo_names(i))
      end do
    else
      write(lunit, '(a)') " none"
    end if

    write(lunit,'(a)') "Scalar fields: "
    if (associated(state%scalar_names)) then
      do i=1,size(state%scalar_names)
        write(lunit,'(a)') " +" // trim(state%scalar_names(i)) &
             // " (" // trim(state%scalar_fields(i)%ptr%name) // ") " &
             // " on " // trim(state%scalar_fields(i)%ptr%mesh%name)
      end do
    else
      write(lunit, '(a)') " none"
    end if

    write(lunit,'(a)') "Vector fields: "
    if (associated(state%vector_names)) then
      do i=1,size(state%vector_names)
        write(lunit,'(a)') " +" // trim(state%vector_names(i)) &
             // " (" // trim(state%vector_fields(i)%ptr%name) // ") " &
             // " on " // trim(state%vector_fields(i)%ptr%mesh%name)
      end do
    else
      write(lunit, '(a)') " none"
    end if

    write(lunit,'(a)') "Tensor fields: "
    if (associated(state%tensor_names)) then
      do i=1,size(state%tensor_names)
        write(lunit,'(a)') " +" // trim(state%tensor_names(i)) &
             // " (" // trim(state%tensor_fields(i)%ptr%name) // ") " &
             // " on " // trim(state%tensor_fields(i)%ptr%mesh%name)
      end do
    else
      write(lunit, '(a)') " none"
    end if

    write(lunit,'(a)') "CSR sparsities: "
    if (associated(state%csr_sparsity_names)) then
      do i=1,size(state%csr_sparsity_names)
        write(lunit,'(a)') " +" // trim(state%csr_sparsity_names(i))
      end do
    else
      write(lunit, '(a)') " none"
    end if

    write(lunit,'(a)') "CSR Matrices: "
    if (associated(state%csr_matrix_names)) then
      do i=1,size(state%csr_matrix_names)
        write(lunit,'(a)') " +" // trim(state%csr_matrix_names(i))
      end do
    else
      write(lunit, '(a)') " none"
    end if

    write(lunit,'(a)') "Block CSR Matrices: "
    if (associated(state%block_csr_matrix_names)) then
      do i=1,size(state%block_csr_matrix_names)
        write(lunit,'(a)') " +" // trim(state%block_csr_matrix_names(i))
      end do
    else
      write(lunit, '(a)') " none"
    end if

  end subroutine print_state

  subroutine select_state_by_mesh(state, mesh_name, mesh_state)
  !!< Returns a state "mesh_state" with only those fields from "state"
  !!< that are defined on a mesh "mesh_name"
    type(state_type), intent(in):: state
    character(len=*), intent(in):: mesh_name
    type(state_type), intent(out):: mesh_state

    type(scalar_field), pointer:: sfield
    type(vector_field), pointer:: vfield
    type(tensor_field), pointer:: tfield
    type(mesh_type), pointer :: old_mesh

    integer j

    call nullify(mesh_state)
    old_mesh => extract_mesh(state, trim(mesh_name))
    call insert(mesh_state, old_mesh, trim(mesh_name))

    ! insert scalar fields defined on "mesh_name"
    do j=1, scalar_field_count(state)
      sfield => extract_scalar_field(state, j)
      if (trim(sfield%mesh%name)==trim(mesh_name)) then
        call insert(mesh_state, sfield, name=trim(sfield%name))
      end if
    end do

    ! insert vector fields defined on "mesh_name"
    do j=1, vector_field_count(state)
      vfield => extract_vector_field(state, j)
      if (trim(vfield%mesh%name)==trim(mesh_name)) then
        call insert(mesh_state, vfield, name=trim(vfield%name))
      end if
    end do

    ! insert tensor fields defined on "mesh_name"
    do j=1, tensor_field_count(state)
      tfield => extract_tensor_field(state, j)
      if (trim(tfield%mesh%name)==trim(mesh_name)) then
        call insert(mesh_state, tfield, name=trim(tfield%name))
      end if
    end do

  end subroutine select_state_by_mesh

  function extract_state(states, name, stat) result (state)
  !!< searches a state by name a returns a pointer to it
    type(state_type), pointer:: state
    type(state_type), dimension(:), intent(in), target:: states
    character(len=*), intent(in):: name
    integer, optional, intent(out):: stat

    integer i

    do i=1, size(states)
      if (states(i)%name==name) exit
    end do

    if (i>size(states)) then
      if (present(stat)) then
        stat=1
        return
      else
        ewrite(-1,*) "Looking for state: "//trim(name)
        FLExit("No such state!")
      end if
    end if

    state => states(i)
    if (present(stat)) stat=0

  end function extract_state

  subroutine collapse_single_state(state, fields)
  !!< Sometimes it is useful to treat everything in state
  !!< as a big bunch of scalar fields -- adapting and
  !!< interpolating spring to mind. Collapse all the fields
  !!< in state down to an array of scalar fields.
    type(state_type), intent(in) :: state
    type(scalar_field), dimension(:), pointer :: fields
    integer :: field, i, j, k, field_count
    type(vector_field), pointer :: field_v
    type(tensor_field), pointer :: field_t

    field_count = scalar_field_count(state)
    do field=1,vector_field_count(state)
      field_v => extract_vector_field(state, field)
      if(trim(field_v%name)=="Coordinate") cycle ! skip Coordinate
      field_count = field_count + field_v%dim
    end do

    do field=1,tensor_field_count(state)
      field_t => extract_tensor_field(state, field)
      field_count = field_count + product(field_t%dim)
    end do

    allocate(fields(field_count))

    i = 1
    do field=1,scalar_field_count(state)
      fields(i) = extract_scalar_field(state, field)
      i = i + 1
    end do

    do field=1,vector_field_count(state)
      field_v => extract_vector_field(state, field)
      if (trim(field_v%name) /= "Coordinate") then
        do j=1,field_v%dim
          fields(i) = extract_scalar_field(field_v, j)
          i = i + 1
        end do
      end if
    end do

    do field=1,tensor_field_count(state)
      field_t => extract_tensor_field(state, field)
      do j=1,field_t%dim(1)
        do k=1,field_t%dim(2)
          fields(i) = extract_scalar_field(field_t, j, k)
          i = i + 1
        end do
      end do
    end do
  end subroutine collapse_single_state

  subroutine collapse_multiple_states(states, fields)
  !!< Sometimes it is useful to treat everything in state
  !!< as a big bunch of scalar fields -- adapting and
  !!< interpolating spring to mind. Collapse all the fields
  !!< in state down to an array of scalar fields.
    type(state_type), dimension(:), intent(in) :: states
    type(scalar_field), dimension(:), pointer :: fields
    integer :: field, i, j, k, field_count
    type(vector_field), pointer :: field_v
    type(tensor_field), pointer :: field_t
    integer :: state

    field_count = 0
    do state=1,size(states)
      field_count = field_count + scalar_field_count(states(state))
      do field=1,vector_field_count(states(state))
        field_v => extract_vector_field(states(state), field)
        if(trim(field_v%name)=="Coordinate") cycle ! skip Coordinate
        field_count = field_count + field_v%dim
      end do

      do field=1,tensor_field_count(states(state))
        field_t => extract_tensor_field(states(state), field)
        field_count = field_count + product(field_t%dim)
      end do
    end do

    allocate(fields(field_count))

    i = 1
    do state=1,size(states)
      do field=1,scalar_field_count(states(state))
        fields(i) = extract_scalar_field(states(state), field)
        i = i + 1
      end do

      do field=1,vector_field_count(states(state))
        field_v => extract_vector_field(states(state), field)
        if (trim(field_v%name) /= "Coordinate") then
          do j=1,field_v%dim
            fields(i) = extract_scalar_field(field_v, j)
            i = i + 1
          end do
        end if
      end do

      do field=1,tensor_field_count(states(state))
        field_t => extract_tensor_field(states(state), field)
        do j=1,field_t%dim(1)
          do k=1,field_t%dim(2)
            fields(i) = extract_scalar_field(field_t, j, k)
            i = i + 1
          end do
        end do
      end do
    end do
  end subroutine collapse_multiple_states

  subroutine collapse_fields_in_single_state(input_state, output_state)
  !!< Sometimes it is useful to treat everything in state
  !!< as a big bunch of scalar fields -- adapting and
  !!< interpolating spring to mind. Collapse all the fields
  !!< in input_state down to scalar fields in output_state.
    type(state_type), intent(in) :: input_state
    type(state_type), intent(out) :: output_state

    type(state_type), dimension(1) :: linput_state, loutput_state

    linput_state = (/input_state/)
    call collapse_fields_in_state(linput_state, loutput_state)
    output_state = loutput_state(1)

  end subroutine collapse_fields_in_single_state

  subroutine collapse_fields_in_multiple_states(input_states, output_states)
  !!< Sometimes it is useful to treat everything in state
  !!< as a big bunch of scalar fields -- adapting and
  !!< interpolating spring to mind. Collapse all the fields
  !!< in input_states down to scalar fields in output_states.
    type(state_type), dimension(:), intent(in) :: input_states
    type(state_type), dimension(:), intent(inout) :: output_states
    integer :: i, j, k, l
    type(scalar_field) :: field_s
    type(vector_field), pointer :: field_v
    type(tensor_field), pointer :: field_t

    assert(size(input_states)==size(output_states))

    do l = 1, size(input_states)
      do i=1,scalar_field_count(input_states(l))
        field_s = extract_scalar_field(input_states(l), i)
        call insert(output_states(l), field_s, trim(input_states(l)%scalar_names(i)))
      end do

      do i=1,vector_field_count(input_states(l))
        field_v => extract_vector_field(input_states(l), i)
        if (trim(field_v%name) /= "Coordinate") then
          do j=1,field_v%dim
            field_s = extract_scalar_field(field_v, j)
            call insert(output_states(l), field_s, &
                        trim(input_states(l)%vector_names(i))//"%"//int2str(j))
          end do
        end if
      end do

      do i=1,tensor_field_count(input_states(l))
        field_t => extract_tensor_field(input_states(l), i)
        do j=1,field_t%dim(1)
          do k=1,field_t%dim(2)
            field_s = extract_scalar_field(field_t, j, k)
            call insert(output_states(l), field_s, &
                        trim(input_states(l)%tensor_names(i))//"%"//int2str((j-1)*field_t%dim(1)+k))
          end do
        end do
      end do
    end do

  end subroutine collapse_fields_in_multiple_states

  function unique_mesh_count(states, seen_ids) result(cnt)
  ! Here we are, reimplementing in an extremely complex manner
  ! something that can be trivially interrogated from spud.
  ! This is stupid.
  type(state_type), intent(in), dimension(:) :: states
  integer :: cnt

  ! We need to have some way of uniquely identifying meshes, so that
  ! we can tell if we've seen this mesh before.
  ! Oh! But Wait! Fortran's hash table support is nonexistent.
  ! So what the blazes are you going to do?
  type(ilist), intent(out), optional :: seen_ids
  type(ilist) :: lseen_ids
  integer :: state, mesh
  type(mesh_type), pointer :: mesh_t

  cnt = 0

  ! This is quadratic. Do you care?
  do state=1,size(states)
    do mesh=1,mesh_count(states(state))
      mesh_t => extract_mesh(states(state), mesh)
      if (.not. has_value(lseen_ids, mesh_t%refcount%id)) then
        cnt = cnt + 1
        call insert(lseen_ids, mesh_t%refcount%id)
      end if
    end do
  end do

  if (present(seen_ids)) then
    seen_ids = lseen_ids
  else
    call deallocate(lseen_ids)
  end if
  end function unique_mesh_count

  subroutine sort_states_by_mesh(states_in, mesh_states)
    type(state_type), intent(in), dimension(:) :: states_in
    type(state_type), dimension(:), allocatable, intent(out) :: mesh_states

    type(ilist) :: seen_ids
    type(inode), pointer :: current_id
    integer :: mesh_count

    integer :: field
    integer :: mesh
    integer :: state
    type(scalar_field), pointer :: sfield
    type(vector_field), pointer :: vfield
    type(tensor_field), pointer :: tfield

    mesh_count = unique_mesh_count(states_in, seen_ids)
    allocate(mesh_states(mesh_count))

    mesh = 0
    current_id => seen_ids%firstnode

    do while(associated(current_id))
      mesh = mesh + 1

      do state=1,size(states_in)
        do field=1,scalar_field_count(states_in(state))
          sfield => extract_scalar_field(states_in(state), field)
          if (sfield%mesh%refcount%id == current_id%value) then
            call insert(mesh_states(mesh), sfield, trim(states_in(state)%name) // trim(sfield%name))
          end if
        end do
        do field=1,vector_field_count(states_in(state))
          vfield => extract_vector_field(states_in(state), field)
          if (vfield%mesh%refcount%id == current_id%value) then
            call insert(mesh_states(mesh), vfield, trim(states_in(state)%name) // trim(vfield%name))
          end if
        end do
        do field=1,tensor_field_count(states_in(state))
          tfield => extract_tensor_field(states_in(state), field)
          if (tfield%mesh%refcount%id == current_id%value) then
            call insert(mesh_states(mesh), tfield, trim(states_in(state)%name) // trim(tfield%name))
          end if
        end do
      end do

      current_id => current_id%next
    end do
    call deallocate(seen_ids)

  end subroutine sort_states_by_mesh

  subroutine halo_update_state(state, level, update_aliased, update_positions)
    !!< Update the halos of fields in the supplied state. If level is not
    !!< supplied, the fields are updated on their largest halo.

    type(state_type), intent(inout) :: state
    integer, optional, intent(in) :: level
    !! If present and false, do *not* update aliased fields
    logical, optional, intent(in) :: update_aliased
    !! If present and true, *do* update the positions field
    logical, optional, intent(in) :: update_positions

    integer :: i
    type(scalar_field), pointer :: s_field => null()
    type(tensor_field), pointer :: t_field => null()
    type(vector_field), pointer :: v_field => null()

    ewrite(2, *) "Updating halos for state " // trim(state%name)

    do i = 1, scalar_field_count(state)
      s_field => extract_scalar_field(state, i)
      if(s_field%field_type == FIELD_TYPE_NORMAL .and. &
        & (.not. present_and_false(update_aliased) .or. &
        & .not. aliased(s_field))) then
        call halo_update(s_field, level = level)
      end if
    end do

    do i = 1, vector_field_count(state)
      v_field => extract_vector_field(state, i)
      if(index(v_field%name,"Coordinate")==len_trim(v_field%name)-9  &
        .and. .not. present_and_true(update_positions)) cycle
      if(v_field%field_type == FIELD_TYPE_NORMAL .and. &
        & (.not. present_and_false(update_aliased) .or. &
        & .not. aliased(v_field))) then
        call halo_update(v_field, level = level)
      end if
    end do

    do i = 1, tensor_field_count(state)
      t_field => extract_tensor_field(state, i)
      if(t_field%field_type == FIELD_TYPE_NORMAL .and. &
        & (.not. present_and_false(update_aliased) .or. &
        & .not. aliased(t_field))) then
        call halo_update(t_field, level = level)
      end if
    end do

  end subroutine halo_update_state

  subroutine halo_update_states(states, level, update_aliased, update_positions)
    !!< Update the halos of fields in the supplied states. If level is not
    !!< supplied, the fields are updated on their largest halo.

    type(state_type), dimension(:), intent(inout) :: states
    integer, optional, intent(in) :: level
    !! If present and true, *do* update aliased fields
    logical, optional, intent(in) :: update_aliased
    !! If present and true, *do* update the positions field
    logical, optional, intent(in) :: update_positions

    integer :: i

    do i = 1, size(states)
      call halo_update(states(i), level = level, update_aliased = present_and_true(update_aliased), update_positions = update_positions)
    end do

  end subroutine halo_update_states

  pure function aliased_scalar(field) result(aliased)
  !!< Checks whether a field is aliased
    !! field to be checked
    type(scalar_field), intent(in) :: field
    logical :: aliased

    aliased=field%aliased

  end function aliased_scalar

  pure function aliased_vector(field) result(aliased)
  !!< Checks whether a field is aliased
    !! field to be checked
    type(vector_field), intent(in) :: field
    logical :: aliased

    aliased=field%aliased

  end function aliased_vector

  pure function aliased_tensor(field) result(aliased)
  !!< Checks whether a field is aliased
    !! field to be checked
    type(tensor_field), intent(in) :: field
    logical :: aliased

    aliased=field%aliased

  end function aliased_tensor

end module state_module
