!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Prof. C Pain
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineeringp
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
module fields_allocates
use elements
use fields_data_types
use fields_base
use shape_functions, only: make_element_shape
use global_parameters, only: PYTHON_FUNC_LEN, empty_path, empty_name, &
     topology_mesh_name, NUM_COLOURINGS, malloc, free
#ifdef __INTEL_COMPILER
use global_parameters, only: c_sizeof
#endif
use halo_data_types
use halos_allocates
use halos_repair
use pickers_deallocates
use adjacency_lists
use global_numbering
use memory_diagnostics
use ieee_arithmetic
use data_structures
use parallel_tools
use python_utils
use iso_c_binding

implicit none

  private

  public :: allocate, deallocate, incref, decref, has_references, add_faces, &
    & deallocate_faces, zero
  public :: make_element_shape, make_mesh, make_mesh_periodic, make_submesh, &
    & create_surface_mesh
  public :: extract_scalar_field, wrap_mesh, wrap_scalar_field, &
    & wrap_tensor_field
  public :: add_lists, extract_lists, add_nnlist, extract_nnlist, add_nelist, &
    & extract_nelist, add_eelist, extract_eelist, remove_lists, remove_nnlist, &
    & remove_nelist, remove_eelist, extract_elements, remove_boundary_conditions, &
    refresh_topology, allocate_region_ids, deallocate_region_ids, add_surface_uid

  interface allocate
     module procedure allocate_scalar_field, allocate_vector_field,&
          & allocate_tensor_field, allocate_mesh, &
          & allocate_scalar_boundary_condition, &
          & allocate_vector_boundary_condition
  end interface

  interface deallocate
     module procedure deallocate_mesh, deallocate_scalar_field,&
          & deallocate_vector_field, deallocate_tensor_field, &
          & deallocate_scalar_boundary_condition, &
          & deallocate_vector_boundary_condition
  end interface

  interface zero
     module procedure zero_scalar, zero_vector, zero_tensor, &
          zero_vector_dim, zero_tensor_dim_dim
  end interface

  interface deallocate_faces
     module procedure deallocate_mesh_faces
  end interface

  interface add_lists
    module procedure add_lists_mesh, add_lists_scalar, add_lists_vector, &
      & add_lists_tensor
  end interface add_lists

  interface extract_lists
    module procedure extract_lists_mesh, extract_lists_scalar, &
      & extract_lists_vector, extract_lists_tensor
  end interface extract_lists

  interface add_nnlist
    module procedure add_nnlist_mesh, add_nnlist_scalar, add_nnlist_vector, &
      & add_nnlist_tensor
  end interface add_nnlist

  interface extract_nnlist
    module procedure extract_nnlist_mesh, extract_nnlist_scalar, &
      & extract_nnlist_vector, extract_nnlist_tensor
  end interface extract_nnlist

  interface add_nelist
    module procedure add_nelist_mesh, add_nelist_scalar, add_nelist_vector, &
      & add_nelist_tensor
  end interface add_nelist

  interface extract_nelist
    module procedure extract_nelist_mesh, extract_nelist_scalar, &
      & extract_nelist_vector, extract_nelist_tensor
  end interface extract_nelist

  interface add_eelist
    module procedure add_eelist_mesh, add_eelist_scalar, add_eelist_vector, &
      & add_eelist_tensor
  end interface add_eelist

  interface extract_eelist
    module procedure extract_eelist_mesh, extract_eelist_scalar, &
      & extract_eelist_vector, extract_eelist_tensor
  end interface extract_eelist

  interface remove_lists
    module procedure remove_lists_mesh
  end interface remove_lists

  interface remove_nnlist
    module procedure remove_nnlist_mesh
  end interface remove_nnlist

  interface remove_nelist
    module procedure remove_nelist_mesh
  end interface remove_nelist

  interface remove_eelist
    module procedure remove_eelist_mesh
  end interface remove_eelist

  interface remove_boundary_conditions
    module procedure remove_boundary_conditions_scalar, &
      remove_boundary_conditions_vector
  end interface remove_boundary_conditions


#include "Reference_count_interface_mesh_type.F90"
#include "Reference_count_interface_scalar_field.F90"
#include "Reference_count_interface_vector_field.F90"
#include "Reference_count_interface_tensor_field.F90"

contains

  subroutine allocate_mesh(mesh, nodes, elements, shape, name)
    type(mesh_type), intent(out) :: mesh
    integer, intent(in) :: nodes, elements
    type(element_type), target, intent(in) :: shape
    character(len=*), intent(in), optional :: name
    integer :: i, j

    mesh%nodes=nodes

    mesh%elements=elements

    mesh%shape=shape
    call incref(shape)

    if (present(name)) then
       mesh%name=name
    else
       mesh%name=empty_name
    end if

    ! should happen in derived type initialisation already,
    ! but just to make sure in case an mesh variable is supplied
    ! that has previously been used for something else:
    nullify(mesh%faces)
    nullify(mesh%columns)
    nullify(mesh%element_columns)

    allocate(mesh%colourings(NUM_COLOURINGS))
    do i = 1, NUM_COLOURINGS
       nullify(mesh%colourings(i)%sets)
    end do
    ! C malloc for interoperability.
    mesh%ndglno_c = malloc(elements*shape%ndof*c_sizeof(1_c_int))
    call c_f_pointer(mesh%ndglno_c, mesh%ndglno, [elements*shape%ndof])

    allocate(mesh%topology)
    mesh%topology%topology => mesh%topology
#ifdef _OPENMP
    ! Use first touch policy.
    !$OMP PARALLEL DO SCHEDULE(STATIC)
    do i=1, mesh%elements
       do j=1, shape%loc
          mesh%ndglno((i-1)*shape%loc+j)=0
       end do
    end do
    !$OMP END PARALLEL DO
#endif

#ifdef HAVE_MEMORY_STATS
    call register_allocation("mesh_type", "integer", elements*shape%ndof,&
         & name=mesh%name)
#endif

    allocate(mesh%adj_lists)
    mesh%wrapped=.false.
    nullify(mesh%region_ids)
    nullify(mesh%subdomain_mesh)
    nullify(mesh%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    mesh%periodic=.false.

    call addref(mesh)
    ! Meshes default to being topology meshes.  Caller should fix this
    ! up later if this is not correct.
    ! This can't use refresh_topology because the topology doesn't
    ! point anywhere yet.
    mesh%topology=mesh

  end subroutine allocate_mesh

  subroutine allocate_scalar_field(field, mesh, name, field_type, py_func, py_positions)
    type(scalar_field), intent(out) :: field
    type(mesh_type), intent(in), target :: mesh
    character(len=*), intent(in),optional :: name
    integer, intent(in), optional :: field_type

    character(len=*), intent(in), optional ::  py_func
    type(vector_field), intent(in), optional, target :: py_positions

    integer :: lfield_type
    integer :: stat
    integer :: toloc, fromloc

    if (present(field_type)) then
       lfield_type = field_type
    else
       lfield_type = FIELD_TYPE_NORMAL
    end if

    field%mesh=mesh
    call incref(mesh)

    if (present(name)) then
       field%name=name
    else
       field%name=empty_name
    end if

    field%field_type = lfield_type
    select case(lfield_type)
    case(FIELD_TYPE_NORMAL)
       allocate(field%val(node_count(mesh)))
       field%py_dim = mesh_dim(mesh)
       field%py_positions_shape => mesh%shape

#ifdef HAVE_MEMORY_STATS
       call register_allocation("scalar_field", "real", node_count(mesh), &
            name=name)
#endif
    case(FIELD_TYPE_CONSTANT)
       allocate(field%val(1))
       field%py_dim = mesh_dim(mesh)
       field%py_positions_shape => mesh%shape

#ifdef HAVE_MEMORY_STATS
       call register_allocation("scalar_field", "real", 1, name=name)
#endif
    case(FIELD_TYPE_DEFERRED)
       allocate(field%val(0))
       field%py_dim = mesh_dim(mesh)
       field%py_positions_shape => mesh%shape
    case(FIELD_TYPE_PYTHON)
       if (present(py_func)) then
          field%py_func = py_func
       else
          if (stat /= 0) then
             FLAbort("Field specified as FIELD_TYPE_PYTHON, but no func passed!")
          end if
       end if

       if (.not. present(py_positions)) then
          FLAbort("Field specified as FIELD_TYPE_PYTHON but no positions field passed!")
       end if
       field%py_positions => py_positions
       field%py_dim = py_positions%dim
       field%py_positions_shape => py_positions%mesh%shape
       call incref(field%py_positions_shape)
       call incref(field%py_positions)

       if (associated(py_positions%mesh%refcount, mesh%refcount)) then
          field%py_positions_same_mesh = .true.
       else
          field%py_positions_same_mesh = .false.
          allocate(field%py_locweight(mesh%shape%ndof, py_positions%mesh%shape%ndof))
          do toloc=1,size(field%py_locweight,1)
             do fromloc=1,size(field%py_locweight,2)
                field%py_locweight(toloc,fromloc)=eval_shape(py_positions%mesh%shape, fromloc, &
                     local_coords(toloc, mesh%shape))
             end do
          end do
       end if

       call add_nelist(field%mesh)
    end select

    field%wrapped=.false.
    field%aliased=.false.
    field%option_path=empty_path
    allocate(field%bc)
    nullify(field%refcount) ! Hacks for gfortran component initialisation
    !                         bug.
    call addref(field)

    call zero(field)

  end subroutine allocate_scalar_field

  subroutine allocate_vector_field(field, dim, mesh, name, field_type)
    type(vector_field), intent(out) :: field
    integer, intent(in) :: dim
    type(mesh_type), intent(in), target :: mesh
    character(len=*), intent(in), optional :: name
    integer, intent(in), optional :: field_type
    integer :: n_count
    integer :: lfield_type

    if (present(field_type)) then
       lfield_type = field_type
    else
       lfield_type = FIELD_TYPE_NORMAL
    end if

    field%dim=dim
    field%option_path=empty_path

    field%mesh=mesh
    call incref(mesh)

    if (present(name)) then
       field%name=name
    else
       field%name=empty_name
    end if

    field%field_type = lfield_type
    select case(lfield_type)
    case(FIELD_TYPE_NORMAL)
       n_count = node_count(mesh)
       call allocate_val(dim,n_count)
#ifdef HAVE_MEMORY_STATS
       call register_allocation("vector_field", "real", n_count*dim, &
            name=name)
#endif
    case(FIELD_TYPE_CONSTANT)
       call allocate_val(dim,1)
#ifdef HAVE_MEMORY_STATS
       call register_allocation("vector_field", "real", dim, name=name)
#endif
    case(FIELD_TYPE_DEFERRED)
       call allocate_val(0,0)
    end select

    field%wrapped = .false.
    field%aliased = .false.
    allocate(field%bc)
    nullify(field%refcount) ! Hack for gfortran component initialisation
    !                         bug.

    allocate(field%picker)

    call addref(field)

    call zero(field)

  contains

    subroutine allocate_val(dim,dofs)
      integer, intent(in) :: dim, dofs

      field%val_c=malloc(dim*dofs*c_sizeof(1.0_c_double))
      call c_f_pointer(field%val_c, field%val, [dim,dofs])

    end subroutine allocate_val

  end subroutine allocate_vector_field

  subroutine allocate_tensor_field(field, mesh, name, field_type, dim)
    type(tensor_field), intent(inout) :: field
    type(mesh_type), intent(in), target :: mesh
    character(len=*), intent(in), optional :: name
    integer, intent(in), optional :: field_type
    integer, intent(in), dimension(2), optional :: dim
    integer :: lfield_type

    if (present(field_type)) then
       lfield_type = field_type
    else
       lfield_type = FIELD_TYPE_NORMAL
    end if

    if(present(dim)) then
       field%dim = dim
    else
       field%dim=(/mesh_dim(mesh),mesh_dim(mesh)/)
    end if
    field%option_path=empty_path

    field%mesh=mesh
    call incref(mesh)

    if (present(name)) then
       field%name=name
    else
       field%name=empty_name
    end if

    field%field_type = lfield_type
    select case(lfield_type)
    case(FIELD_TYPE_NORMAL)
       allocate(field%val(field%dim(1), field%dim(2), node_count(mesh)))

#ifdef HAVE_MEMORY_STATS
       call register_allocation("tensor_field", "real", &
            node_count(mesh)*field%dim(1)*field%dim(2), name=name)
#endif
    case(FIELD_TYPE_CONSTANT)
       allocate(field%val(field%dim(1), field%dim(2), 1))

#ifdef HAVE_MEMORY_STATS
       call register_allocation("tensor_field", "real", &
            field%dim(1)*field%dim(2), name=name)
#endif
    case(FIELD_TYPE_DEFERRED)
       allocate(field%val(0, 0, 0))
    end select

    field%wrapped=.false.
    field%aliased=.false.
    nullify(field%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    call addref(field)

    call zero(field)

  end subroutine allocate_tensor_field

  subroutine deallocate_subdomain_mesh(mesh)
    type(mesh_type) :: mesh

    if (.not.associated(mesh%subdomain_mesh)) return

    deallocate(mesh%subdomain_mesh%element_list)
    deallocate(mesh%subdomain_mesh%node_list)

    deallocate(mesh%subdomain_mesh)

  end subroutine deallocate_subdomain_mesh

  subroutine deallocate_mesh_faces(mesh)
    type(mesh_type) :: mesh

    if (.not.associated(mesh%faces)) return

    call deallocate(mesh%faces%face_list)

#ifdef HAVE_MEMORY_STATS
    call register_deallocation("mesh_type", "integer", &
         size(mesh%faces%face_lno), name=mesh%name)
#endif
    deallocate(mesh%faces%face_lno)

#ifdef HAVE_MEMORY_STATS
    call register_deallocation("mesh_type", "integer", &
         size(mesh%faces%face_element_list), &
         name=mesh%name)
#endif
    deallocate(mesh%faces%face_element_list)
#ifdef HAVE_MEMORY_STATS
    call register_deallocation("mesh_type", "integer", &
         size(mesh%faces%local_face_number), &
         name=mesh%name)
#endif
    deallocate(mesh%faces%local_face_number)

    call deallocate(mesh%faces%shape%quadrature)
    call deallocate(mesh%faces%shape)
    deallocate(mesh%faces%shape)

    call deallocate(mesh%faces%surface_mesh)

#ifdef HAVE_MEMORY_STATS
    call register_deallocation("mesh_type", "integer", &
         size(mesh%faces%surface_node_list), name=trim(mesh%name)//" surface_nodes")
#endif
    if (associated(mesh%faces%surface_node_list)) then
       deallocate(mesh%faces%surface_node_list)
    end if

#ifdef HAVE_MEMORY_STATS
    call register_deallocation("mesh_type", "integer", &
         size(mesh%faces%boundary_ids), &
         name=trim(mesh%name)//" boundary_ids")
#endif
    call free(mesh%faces%boundary_ids_c)
    mesh%faces%boundary_ids => null()

    if (.not.c_associated(mesh%faces%interior_local_facet_number)) then
       call free(mesh%faces%interior_local_facet_number)
    end if
    if (.not.c_associated(mesh%faces%exterior_local_facet_number)) then
       call free(mesh%faces%exterior_local_facet_number)
    end if
    if (.not.c_associated(mesh%faces%exterior_facet_cell)) then
       call free(mesh%faces%exterior_facet_cell)
    end if
    if (.not.c_associated(mesh%faces%interior_facet_cell)) then
       call free(mesh%faces%interior_facet_cell)
    end if

    if (associated(mesh%faces%coplanar_ids)) then
       deallocate(mesh%faces%coplanar_ids)
    end if

    if (associated(mesh%faces%dg_surface_mesh)) then
       call deallocate(mesh%faces%dg_surface_mesh)
       deallocate(mesh%faces%dg_surface_mesh)
    end if

    deallocate(mesh%faces)

  end subroutine deallocate_mesh_faces

  subroutine deallocate_mesh(mesh)
    !!< Deallocate the components of mesh. Shape functions are not
    !!< deallocated here.
    type(mesh_type), intent(inout) :: mesh
    integer :: i, uid

    ! Save the uid
    uid = mesh%refcount%id

    call decref(mesh)
    if (has_references(mesh)) then
       ! There are still references to this mesh so we don't deallocate.
       return
    end if

    call python_remove_from_cache("mesh", uid)

    call deallocate(mesh%shape)

    if (.not.mesh%wrapped) then
#ifdef HAVE_MEMORY_STATS
       call register_deallocation("mesh_type", "integer", &
            size(mesh%ndglno), name=mesh%name)
#endif
       call free(mesh%ndglno_c)
       mesh%ndglno_c=C_NULL_PTR
       mesh%ndglno=>null()
    end if

    if(associated(mesh%region_ids)) then
       call deallocate_region_ids(mesh)
    end if

    assert(associated(mesh%adj_lists))
    call remove_lists(mesh)
    deallocate(mesh%adj_lists)
    nullify(mesh%adj_lists)

    if(associated(mesh%halos)) then
       call deallocate(mesh%halos)
       deallocate(mesh%halos)
    end if

    if(associated(mesh%element_halos)) then
       call deallocate(mesh%element_halos)
       deallocate(mesh%element_halos)
    end if

    call deallocate_faces(mesh)

    if (associated(mesh%topology)) then
       deallocate(mesh%topology)
       nullify(mesh%topology)
    end if

    if(associated(mesh%subdomain_mesh)) then
       call deallocate_subdomain_mesh(mesh)
    end if

    if(associated(mesh%columns)) then
       deallocate(mesh%columns)
    end if

    if(associated(mesh%element_columns)) then
       deallocate(mesh%element_columns)
    end if

    if(associated(mesh%colourings)) then
       do i = 1, NUM_COLOURINGS
          if(associated(mesh%colourings(i)%sets)) then
             call deallocate(mesh%colourings(i)%sets)
             deallocate(mesh%colourings(i)%sets)
          end if
       end do
       deallocate(mesh%colourings)
    end if

    if (.not.c_associated(mesh%interior_facet_dof_list)) then
       call free(mesh%interior_facet_dof_list)
    end if
    if (.not.c_associated(mesh%exterior_facet_dof_list)) then
       call free(mesh%exterior_facet_dof_list)
    end if

  end subroutine deallocate_mesh

  recursive subroutine deallocate_scalar_field(field)
    !!< Deallocate the storage associated with the field values. Deallocate
    !!< is called on the mesh which will delete one reference to it and
    !!< deallocate it if the count drops to zero.
    type(scalar_field), intent(inout) :: field

    integer uid

    ! Save the uid
    uid = field%refcount%id

    call decref(field)
    if (has_references(field)) then
       ! There are still references to this field so we don't deallocate.
       return
    end if

    call python_remove_from_cache("scalar_field", uid)

    select case(field%field_type)
    case(FIELD_TYPE_NORMAL)
       if (.not.field%wrapped) then
#ifdef HAVE_MEMORY_STATS
          call register_deallocation("scalar_field", "real", &
               size(field%val), name=field%name)
#endif

#ifdef DDEBUG
          field%val = ieee_value(0.0, ieee_quiet_nan)
#endif
          deallocate(field%val)
       end if
    case(FIELD_TYPE_CONSTANT)
#ifdef HAVE_MEMORY_STATS
       call register_deallocation("scalar_field", "real", &
            1, name=field%name)
#endif

#ifdef DDEBUG
       field%val = ieee_value(0.0, ieee_quiet_nan)
#endif
       deallocate(field%val)
    case(FIELD_TYPE_PYTHON)
       call deallocate(field%py_positions)
       call deallocate(field%py_positions_shape)
       if (associated(field%py_locweight)) then
          deallocate(field%py_locweight)
       end if
    case(FIELD_TYPE_DEFERRED)
       FLAbort("You were supposed to allocate the deferred field later!")
    end select

    call deallocate(field%mesh)

    call remove_boundary_conditions(field)
    deallocate(field%bc)

  end subroutine deallocate_scalar_field

  subroutine remove_boundary_conditions_scalar(field)
    !!< Removes and deallocates all boundary conditions from a field
    type(scalar_field), intent(inout):: field

    integer:: i

    if (associated(field%bc%boundary_condition)) then
       do i=1, size(field%bc%boundary_condition)
          call deallocate(field%bc%boundary_condition(i))
       end do
       deallocate(field%bc%boundary_condition)
    end if

  end subroutine remove_boundary_conditions_scalar

  recursive subroutine deallocate_vector_field(field)
    !!< Deallocate the storage associated with the field values. Deallocate
    !!< is called on the mesh which will delete one reference to it and
    !!< deallocate it if the count drops to zero.
    type(vector_field), intent(inout) :: field

    integer uid

    ! Save the uid
    uid = field%refcount%id

    call decref(field)
    if (has_references(field)) then
       ! There are still references to this field so we don't deallocate.
       return
    end if

    call python_remove_from_cache("vector_field", uid)

    if (.not.field%wrapped) then
       select case(field%field_type)
       case(FIELD_TYPE_NORMAL,FIELD_TYPE_CONSTANT)
#ifdef DDEBUG
          field%val = ieee_value(0.0, ieee_quiet_nan)
#endif
#ifdef HAVE_MEMORY_STATS
          call register_deallocation("vector_field", "real", &
               size(field%val), name=field%name)
#endif
          call free(field%val_c)
          field%val_c = C_NULL_PTR
          field%val=>null()
       case(FIELD_TYPE_DEFERRED)
          FLAbort("You were supposed to allocate the deferred field later!")
       end select
    end if

    call deallocate(field%mesh)

    call remove_boundary_conditions(field)
    deallocate(field%bc)

    assert(associated(field%picker))
    call remove_picker(field)
    deallocate(field%picker)
    nullify(field%picker)

  end subroutine deallocate_vector_field

  subroutine remove_boundary_conditions_vector(field)
    !!< Removes and deallocates all boundary conditions from a field
    type(vector_field), intent(inout):: field

    integer:: i

    if (associated(field%bc%boundary_condition)) then
       do i=1, size(field%bc%boundary_condition)
          call deallocate(field%bc%boundary_condition(i))
       end do
       deallocate(field%bc%boundary_condition)
    end if

  end subroutine remove_boundary_conditions_vector

  subroutine deallocate_tensor_field(field)
    !!< Deallocate the storage associated with the field values. Deallocate
    !!< is called on the mesh which will delete one reference to it and
    !!< deallocate it if the count drops to zero.
    type(tensor_field), intent(inout) :: field

    integer uid

    ! Save the uid
    uid = field%refcount%id

    call decref(field)
    if (has_references(field)) then
       ! There are still references to this field so we don't deallocate.
       return
    end if

    call python_remove_from_cache("tensor_field", uid)

    if (.not.field%wrapped) then
       select case(field%field_type)
       case(FIELD_TYPE_NORMAL,FIELD_TYPE_CONSTANT)
#ifdef HAVE_MEMORY_STATS
          call register_deallocation("tensor_field", "real", &
               size(field%val), field%name)
#endif

#ifdef DDEBUG
          field%val = ieee_value(0.0, ieee_quiet_nan)
#endif
          deallocate(field%val)
       case(FIELD_TYPE_DEFERRED)
          FLAbort("You were supposed to allocate the deferred field later!")
       end select
    end if

    call deallocate(field%mesh)

  end subroutine deallocate_tensor_field

  subroutine allocate_scalar_boundary_condition(bc, mesh, surface_element_list, &
       name, type)
    !!< Allocate a scalar boundary condition
    type(scalar_boundary_condition), intent(out):: bc
    type(mesh_type), intent(in):: mesh
    !! surface elements to which this b.c. applies (is copied in)
    integer, dimension(:), intent(in):: surface_element_list
    !! all things should have a name
    character(len=*), intent(in):: name
    !! type can be any of: ...
    character(len=*), intent(in):: type

    bc%name=name
    bc%type=type
    allocate( bc%surface_element_list(1:size(surface_element_list)) )
    bc%surface_element_list=surface_element_list
    allocate(bc%surface_mesh)
    call create_surface_mesh(bc%surface_mesh, bc%surface_node_list, &
         mesh, bc%surface_element_list, name=trim(name)//'Mesh')

  end subroutine allocate_scalar_boundary_condition

  subroutine allocate_vector_boundary_condition(bc, mesh, surface_element_list, &
       applies, name, type)
    !!< Allocate a vector boundary condition
    type(vector_boundary_condition), intent(out):: bc
    type(mesh_type), intent(in):: mesh
    !! surface elements of this mesh to which this b.c. applies (is copied in):
    integer, dimension(:), intent(in):: surface_element_list
    !! all things should have a name
    character(len=*), intent(in):: name
    !! type can be any of: ...
    character(len=*), intent(in):: type
    !! b.c. only applies for components with applies==.true.
    logical, dimension(:), intent(in), optional:: applies

    bc%name=name
    bc%type=type
    allocate( bc%surface_element_list(1:size(surface_element_list)) )
    bc%surface_element_list=surface_element_list
    allocate(bc%surface_mesh)
    call create_surface_mesh(bc%surface_mesh, bc%surface_node_list, &
         mesh, bc%surface_element_list, name=trim(name)//'Mesh')

    if (present(applies)) then
       ! size(bc%applies) is always 3! also for dim<3
       bc%applies(1:size(applies))=applies
       bc%applies(size(applies)+1:)=.false.
    else
       ! default .true. for all components
       bc%applies=.true.
    end if

  end subroutine allocate_vector_boundary_condition

  subroutine deallocate_scalar_boundary_condition(bc)
    !! deallocate a scalar boundary condition
    type(scalar_boundary_condition), intent(inout):: bc

    integer i

    if (associated(bc%surface_fields)) then
       do i=1, size(bc%surface_fields)
          call deallocate(bc%surface_fields(i))
       end do
       deallocate(bc%surface_fields)
    end if

    call deallocate(bc%surface_mesh)
    deallocate(bc%surface_mesh)

    deallocate(bc%surface_element_list, bc%surface_node_list)

  end subroutine deallocate_scalar_boundary_condition

  subroutine deallocate_vector_boundary_condition(bc)
    !! deallocate a vector boundary condition
    type(vector_boundary_condition), intent(inout):: bc

    integer i

    if (associated(bc%surface_fields)) then
       do i=1, size(bc%surface_fields)
          call deallocate(bc%surface_fields(i))
       end do
       deallocate(bc%surface_fields)
    end if

    if (associated(bc%scalar_surface_fields)) then
       do i=1, size(bc%scalar_surface_fields)
          call deallocate(bc%scalar_surface_fields(i))
       end do
       deallocate(bc%scalar_surface_fields)
    end if

    call deallocate(bc%surface_mesh)
    deallocate(bc%surface_mesh)

    deallocate(bc%surface_element_list, bc%surface_node_list)
  end subroutine deallocate_vector_boundary_condition

  !---------------------------------------------------------------------
  ! routines for wrapping meshes and fields around provided arrays
  !---------------------------------------------------------------------

  function wrap_mesh(ndglno, shape, name) result (mesh)
    !!< Return a mesh wrapped around the information provided.
    type(mesh_type) :: mesh

    integer, dimension(:), target, intent(in) :: ndglno
    type(element_type), target, intent(in) :: shape
    character(len=*), intent(in) :: name

    mesh%ndglno=>ndglno
    mesh%shape=shape
    call incref(shape)
    nullify(mesh%faces)

    mesh%name=name

    mesh%elements=size(ndglno)/shape%ndof

    allocate(mesh%adj_lists)
    mesh%wrapped=.true.
    mesh%nodes=maxval(ndglno)
    nullify(mesh%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    mesh%periodic = .false. ! can only really assume that this is false as
    ! we have no other information
    call addref(mesh)

  end function wrap_mesh

  function wrap_scalar_field(mesh, val, name, val_stride) result (field)
    !!< Return a scalar field wrapped around the arrays provided.
    type(scalar_field) :: field

    type(mesh_type), target, intent(in) :: mesh
    real, dimension(:), target, intent(in) :: val
    character(len=*), intent(in) :: name
    !! has to be provided if the val array is non-contiguous in memory!
    integer, optional:: val_stride

    field%val=>val
    field%mesh=mesh

    field%name=name
    if (present(val_stride)) then
       field%val_stride=val_stride
    else
       field%val_stride=1
    end if

    field%py_dim = mesh_dim(mesh)
    field%py_positions_shape => mesh%shape

    field%wrapped = .true.
    call incref(mesh)
    allocate(field%bc)
    nullify(field%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    call addref(field)

  end function wrap_scalar_field

  function wrap_tensor_field(mesh, val, name) result (field)
    !!< Return a tensor field wrapped around the arrays provided.
    type(tensor_field) :: field

    type(mesh_type), target, intent(in) :: mesh
    real, dimension(mesh_dim(mesh), mesh_dim(mesh), node_count(mesh)),&
         & target, intent(in) :: val
    character(len=*), intent(in) :: name

    field%val=>val
    field%mesh=mesh
    field%dim=mesh_dim(mesh)

    field%name=name

    field%wrapped=.true.
    call incref(mesh)
    nullify(field%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    call addref(field)

  end function wrap_tensor_field

  function make_mesh (model, shape, continuity, name, with_faces, dofs_per_column) &
       result (mesh)
    !!< Produce a mesh based on an old mesh but with a different shape and/or continuity.
    type(mesh_type) :: mesh

    type(mesh_type), intent(in), target :: model
    type(element_type), target, intent(in), optional :: shape
    integer, intent(in), optional :: continuity
    character(len=*), intent(in), optional :: name
    logical, intent(in), optional :: with_faces
    integer, dimension(0:), intent(in), optional :: dofs_per_column

    integer, dimension(:), allocatable :: ndglno
    real, dimension(:), pointer :: val
    integer :: i, j, input_nodes, n_faces
    type(element_type), pointer :: inshape
    type(element_type) :: lshape

    if (present(shape)) then
       inshape=>shape
    else
       inshape=>model%shape
    end if

    if (present(continuity)) then
       if (continuity<0.and.inshape%type==ELEMENT_LAGRANGIAN) then
          lshape = make_element_shape(inshape, type=ELEMENT_DISCONTINUOUS_LAGRANGIAN)
       else
          lshape=inshape
          call incref(lshape)
       end if
    else
       lshape=inshape
       call incref(lshape)
    end if

    call allocate_mesh(mesh, &
         nodes=0, & ! We'll fix this later.
         elements=model%elements, &
         shape=lshape, &
         name=name)

    if (any(lshape%type==(/ELEMENT_TRACE, &
         ELEMENT_DISCONTINUOUS_LAGRANGIAN/))) then
       mesh%continuity=-1
    else
       mesh%continuity=0
    end if

    mesh%topology=mesh_topology(model)
    mesh%uid=>model%uid

    mesh%periodic=model%periodic

    ! You can't have a CG degree 0 mesh!
    ! Fluidity doesn't know about outer product elements (lshape%type
    ! == -666) so complain.
    if(mesh%shape%degree==0.and.mesh%continuity>=0.and.mesh%shape&
         &%type/=ELEMENT_TRACE.and.lshape%type/=-666) then
       FLExit("For a P0 mesh, the 'mesh_continuity' must be Discontinuous.")
    end if

    if (associated(model%region_ids)) then
       call allocate_region_ids(mesh, size(model%region_ids))
       mesh%region_ids=model%region_ids
    end if

    mesh%element_classes = model%element_classes

    call make_global_numbering_new(mesh, dofs_per_column=dofs_per_column)

    if (associated(model%element_halos)) then
       assert(element_halo_count(model) > 0)

       allocate(mesh%element_halos(size(model%element_halos)))
       do i=1,size(mesh%element_halos)
          mesh%element_halos(i)=model%element_halos(i)
          call incref(mesh%element_halos(i))
       end do

    end if

    call deallocate(lshape)

    ! Transfer the eelist from model to mesh
    assert(associated(model%adj_lists))
    if(associated(model%adj_lists%eelist)) then
       ewrite(2, *) "Transferring element-element list to mesh " // trim(mesh%name)
       allocate(mesh%adj_lists%eelist)
       mesh%adj_lists%eelist = model%adj_lists%eelist
       call incref(mesh%adj_lists%eelist)
    end if

    call refresh_topology(mesh)

    if(has_faces(model).and..not.present_and_false(with_faces)) then
       call add_faces(mesh, model)
    end if
  end function make_mesh

  subroutine add_faces(mesh, model, sndgln, sngi, boundary_ids, &
       periodic_face_map, element_owner, incomplete_surface_mesh, stat)
    !!< Subroutine to add a faces component to mesh. Since mesh may be
    !!< discontinuous, a continuous model mesh must
    !!< be provided. To avoid duplicate computations, and ensure
    !!< consistent numbering one should first call add_faces on the model
    !!< mesh. If no model is provided, the mesh must be continuous.
    !!< WARNING: after the model mesh is deallocated the faces component
    !!< of 'mesh' also becomes invalid!
    type(mesh_type), target :: mesh
    !!< model is only changed when periodic_face_map is provided (see below)
    !!< not using intent(inout) - which is allowed as we only change pointed to values
    type(mesh_type), optional, target, intent(in) :: model
    !! surface mesh (ndglno using the same node numbering as in 'mesh')
    !! if present the N elements in this mesh will correspond to the first
    !! N faces of the new faces component.
    !! LEGACY: Any other faces found to be on the boundary of the domain are
    !! inserted after that. So that with M=surface_element_count(mesh) (where M>=N)
    !! the first 1:M faces form a complete surface mesh of the domain. This
    !! is legacy functionality that only works in serial.
    !! A warning will therefore be issued.
    !! This legacy behaviour can be switched off with
    !! incomplete_surface_mesh=.true. (see below)
    integer, dimension(:), intent(in), optional:: sndgln
    !! number of quadrature points for the faces mesh (if not present the
    !! degree of 'mesh' is maintained):
    integer, intent(in), optional:: sngi
    !! A list of ids marking different parts of the surface mesh
    !! so that boundary conditions can be associated with it.
    integer, dimension(:), intent(in), optional :: boundary_ids
    !! if supplied the mesh is periodic and the model non-periodic
    !! this list must contain the pairs of faces, on the periodic boundary,
    !! that are now between two elements. This is needed to update the face_list
    !! Additionaly the face local node numbering of the *model* mesh is updated
    !! be consistent with that of the periodic face local node numbering.
    type(integer_hash_table), intent(in), optional :: periodic_face_map
    !! This list gives the element owning each face in sndgln. This needs
    !! to be provided when sndgln contains internal faces, as we need to
    !! decide which of the two adjacent elements the face belongs to
    !! (used in reading in periodic meshes which include the periodic faces in the surface mesh)
    integer, dimension(:), intent(in), optional :: element_owner
    !! See comments above sndgln
    logical, intent(in), optional :: incomplete_surface_mesh
    integer, intent(out), optional :: stat

    type(integer_hash_table):: lperiodic_face_map
    type(mesh_type), pointer :: lmodel
    type(element_type), pointer :: element
    type(quadrature_type) :: quad_face
    integer, dimension(:), pointer :: faces, neigh, model_ele_glno, model_ele_glno2
    integer, dimension(1:mesh%shape%cell%entity_counts(0)) :: vertices, &
         ele_facet, ele_facet2 ! these last two are actually smaller
    integer :: face_count, ele, j, snloc, m, n, p, face2
    integer, dimension(2) :: local_facet

    if (present(stat)) then
       stat = 0
    end if

    if (associated(mesh%faces)) then
       ! calling add_faces twice is dangerous as it may nuke information
       ! supplied in the first call (such as sndgln, boundary_ids)
       if (present(stat)) then
          stat = 1
          return
       else
          ewrite(0,*) "add_faces is already called for this mesh"
          ewrite(0,*) "call deallocate_faces() first if you want to recompute"
          FLAbort("The end.")
       end if
    end if

    allocate(mesh%faces)
    call refresh_topology(mesh)
    ! only created in the first call to get_dg_surface_mesh()
    mesh%faces%dg_surface_mesh => null()

    if (.not. present(model)) then

       ! create mesh%faces%face_list an integer csr matrix storing the
       !     face number between each (directed) pair of adjacent elements
       !     note that this is an asymmetric matrix A
       !     where A_ij gives the facet of element i facing element j
       !       and A_ji the facet of element j facing element i
       !       (i.e. there are 2 opposite faces between two elements)
       ! and mesh%faces%face_element_list  storing the element adjacent to
       !     each face
       call add_faces_face_list(mesh, sndgln, &
            boundary_ids=boundary_ids, &
            element_owner=element_owner, &
            incomplete_surface_mesh=incomplete_surface_mesh)

       ! we don't calculate coplanar_ids here, as we need positions
       mesh%faces%coplanar_ids => null()

       lmodel => mesh

    else

       ! mesh%faces%face_list and mesh%faces%face_element_list
       ! are the same as for the model, so are simply copied
       ! (unless periodic)
       assert(continuity(model)>=0)

       if (.not. associated(model%faces)) then
          FLAbort("One should call add_faces on the model mesh first.")
       end if

       lmodel => model

       if (present(periodic_face_map)) then

          ! make face_list from the model but change periodic faces to become internal
          call add_faces_face_list_periodic_from_non_periodic_model( &
               mesh, model, periodic_face_map)

          ! Having fixed the face list, we should now use the original mesh
          ! rather than the model so that all faces which are supposed to be
          ! adjacent actually are.
          lmodel=>mesh
          ! works as long as we're not discontinuous
          assert( mesh%continuity>=0 )

          ! the periodic faces will be internal faces in the output periodic mesh
          mesh%faces%has_internal_boundaries = .true.

       else if (model%periodic .and. .not. mesh%periodic) then

          ! make face_list from the model but change periodic faces to normal external faces
          call add_faces_face_list_non_periodic_from_periodic_model( &
               mesh, model, lperiodic_face_map, stat=stat)

          ! the subroutine above only works if the removing of periodic bcs has removed all internal boundaries
          mesh%faces%has_internal_boundaries = .true.

       else
          ! Transfer the faces from model to mesh
          mesh%faces%face_list=model%faces%face_list
          call incref(mesh%faces%face_list)

          ! have internal faces if the model does
          mesh%faces%has_internal_boundaries = has_internal_boundaries(model)
       end if

       ! face_element_list is a pure copy of that of the model
       allocate( mesh%faces%face_element_list(1:size(model%faces%face_element_list)) )
       mesh%faces%face_element_list=model%faces%face_element_list
#ifdef HAVE_MEMORY_STATS
       call register_allocation("mesh_type", "integer", &
            size(mesh%faces%face_element_list), &
            trim(mesh%name)//" face_element_list.")
#endif

       ! local_face_number is a pure copy of that of the model
       allocate( mesh%faces%local_face_number(1:size(model%faces%local_face_number)) )
       mesh%faces%local_face_number=model%faces%local_face_number
#ifdef HAVE_MEMORY_STATS
       call register_allocation("mesh_type", "integer", &
            size(mesh%faces%local_face_number), &
            trim(mesh%name)//" local_face_number.")
#endif

       ! boundary_ids is a pure copy of that of model
       mesh%faces%boundary_ids_c = malloc(size(model%faces%boundary_ids) *&
            & c_sizeof(1_c_int))
       call c_f_pointer(mesh%faces%boundary_ids_c, mesh%faces%boundary_ids,&
            & [size(model%faces%boundary_ids)])

       mesh%faces%boundary_ids=model%faces%boundary_ids
#ifdef HAVE_MEMORY_STATS
       call register_allocation("mesh_type", "integer", &
            size(mesh%faces%boundary_ids), &
            trim(mesh%name)//" boundary_ids")
#endif

       ! same for coplanar ids (if existent)
       if (associated(model%faces%coplanar_ids)) then
          allocate( mesh%faces%coplanar_ids(1:size(model%faces%coplanar_ids)) )
          mesh%faces%coplanar_ids=model%faces%coplanar_ids
       else
          nullify(mesh%faces%coplanar_ids)
       end if

    end if ! if (.not. present(model)) then ... else ...

    ! at this point mesh%faces%face_list and mesh%faces%face_element_list are
    ! ready (either newly computed or copied from model)
    ! now we only have to work out mesh%faces%face_lno

    element => ele_shape(mesh, 1)
    if (present(sngi)) then
       ! if specified use quadrature with sngi gausspoints
       quad_face = make_quadrature(vertices=face_vertices(element), &
            & dim=mesh_dim(mesh)-1, ngi=sngi, family=element%quadrature%family)
       ! quad_face will be deallocated inside deallocate_faces!
    else
       ! otherwise use degree of full mesh
       quad_face = make_quadrature(vertices=face_vertices(element), &
            & dim=mesh_dim(mesh)-1, degree=element%quadrature%degree, family=element%quadrature%family)
       ! quad_face will be deallocated inside deallocate_faces!
    end if

    allocate(mesh%faces%shape)
    mesh%faces%shape = make_element_shape(vertices=face_vertices(element), &
         & dim=mesh_dim(mesh)-1, degree=element%degree, quad=quad_face)

    !FIXME THIS IS COPIED FROM THE TRUNK AND IS ALMOST CERTAINLY NOW WRONG
    face_count=entries(mesh%faces%face_list)
    snloc=mesh%faces%shape%ndof
    allocate(mesh%faces%face_lno( face_count*snloc ))
#ifdef HAVE_MEMORY_STATS
    call register_allocation("mesh_type", "integer", &
         size(mesh%faces%face_lno), name=mesh%name)
#endif

    vertices=local_vertices(lmodel%shape%numbering)

    ! now fill in face_lno
    eleloop: do ele=1, size(mesh%faces%face_list,1)

       faces => row_ival_ptr(mesh%faces%face_list, ele)
       neigh => row_m_ptr(mesh%faces%face_list, ele)
       model_ele_glno => ele_nodes(lmodel, ele)

       faceloop: do j=1, size(faces)
          if (ele<neigh(j)) then
             ! interior face between ele and neigh(j)
             ! ele<neigh(j) to ensure each pair {ele, neigh(j)} is handled once

             model_ele_glno2 => ele_nodes(lmodel, neigh(j))
             p=0
             ! Look for common boundaries by matching common vertices
             ! Note that we have to use the model mesh here
             do m=1,size(vertices)
                do n=1,size(vertices)
                   if (model_ele_glno(vertices(m))==model_ele_glno2(vertices(n))) then
                      p=p+1
                      ele_facet(p)=m
                      ele_facet2(p)=n
                   end if
                end do
             end do

             ! Check that we really have found two boundaries.
             ASSERT(p==vertex_count(lmodel%faces%shape))
             ! (this might break for the case where elements share more than one
             ! face, but in that case the next few lines are wrong as well)

             mesh%faces%face_lno((faces(j)-1)*snloc+1:faces(j)*snloc)= &
                  sorted_facet(mesh%shape, j, ele_facet(1:p))

             face2=ival(mesh%faces%face_list, neigh(j), ele)

             local_facet=vertices_entity(mesh%shape%cell,&
                  & sorted(ele_facet2(1:p)))

             mesh%faces%face_lno((face2-1)*snloc+1:face2*snloc)= &
                  sorted_facet(mesh%shape, local_facet(2), ele_facet2(1:p))

          else if (neigh(j)<0) then

             ! boundary face:
             mesh%faces%face_lno((faces(j)-1)*snloc+1:faces(j)*snloc)= &
                  & mesh%shape%facet2dofs(j)%dofs

          end if

       end do faceloop
    end do eleloop

    if (present(periodic_face_map)) then
       !call fix_periodic_face_orientation(model, mesh, periodic_face_map)
    else if (present(model)) then
       if (model%periodic .and. .not. mesh%periodic) then
          nullify(mesh%faces%surface_node_list)
          !call fix_periodic_face_orientation(mesh, model, lperiodic_face_map)
          call deallocate(lperiodic_face_map)
       end if
    end if

    ! this is a surface mesh consisting of all exterior faces
    !    which is often used and therefore created in advance
    ! this also create a surface_node_list which can be used
    !    as a mapping between the node numbering of this surface mesh
    !    and the node numbering of the full mesh
    call create_surface_mesh(mesh%faces%surface_mesh, &
         mesh%faces%surface_node_list, mesh, name='Surface'//trim(mesh%name))

#ifdef HAVE_MEMORY_STATS
    call register_allocation("mesh_type", "integer", &
         size(mesh%faces%surface_node_list), name='Surface'//trim(mesh%name))
#endif

  end subroutine add_faces

  subroutine add_faces_face_list(mesh, sndgln, boundary_ids, &
       element_owner, incomplete_surface_mesh)
    !!< Subroutine to calculate the face_list and face_element_list of the
    !!< faces component of a 'model mesh'. This should be the linear continuous
    !!< mesh that may serve as a 'model' for other meshes.
    type(mesh_type), intent(inout) :: mesh
    !! surface mesh (ndglno using the same node numbering as in 'mesh')
    !! if present the N elements in this mesh will correspond to the first
    !! N faces of the new faces component.
    integer, dimension(:), target, intent(in), optional:: sndgln
    integer, dimension(:), target, intent(in), optional:: boundary_ids
    integer, dimension(:), intent(in), optional :: element_owner
    logical, intent(in), optional :: incomplete_surface_mesh

    type(element_type), pointer :: mesh_shape
    type(element_type) :: face_shape
    logical:: internal_face, surface_elements_added
    ! listen very carefully, I shall say this only once:
    logical, save :: warning_given=.false.
    integer, dimension(:), pointer :: faces, neigh, snodes
    integer, dimension(2) :: common_elements
    integer :: nloc, snloc, stotel
    integer :: bdry_count, ele, sele, j, no_found
    integer :: no_faces
    type(csr_sparsity) :: sparsity
    type(csr_sparsity) :: nelist

    assert(continuity(mesh)>=0)

    mesh_shape=>ele_shape(mesh, 1)
    nloc=mesh_shape%ndof

    ! Calculate the node-to-element list.
    ! Calculate the element adjacency list.
    call extract_lists(mesh, nelist = nelist, eelist = sparsity)

    call allocate(mesh%faces%face_list, sparsity, type=CSR_INTEGER, &
         name=trim(mesh%name)//"FaceList")
    call zero(mesh%faces%face_list)

    no_faces=size(mesh%faces%face_list%sparsity%colm)
    allocate(mesh%faces%face_element_list(no_faces))
#ifdef HAVE_MEMORY_STATS
    call register_allocation("mesh_type", "integer", no_faces, &
         trim(mesh%name)//" face_element_list")
#endif
    allocate(mesh%faces%local_face_number(no_faces))
#ifdef HAVE_MEMORY_STATS
    call register_allocation("mesh_type", "integer", no_faces, &
         trim(mesh%name)//" local_face_number")
#endif

    mesh%faces%has_internal_boundaries = present(element_owner)

    snloc=face_vertices(mesh_shape)
    if (present(sndgln)) then

       stotel=size(sndgln)/snloc

       do sele=1, stotel

          ! find the elements adjacent to this surface element
          snodes => sndgln( (sele-1)*snloc+1:sele*snloc )
          call FindCommonElements(common_elements, no_found, nelist, &
               nodes=snodes )

          if (no_found==1) then
             ! we have found the adjacent element
             ele=common_elements(1)
             internal_face=.false.
             if (present(element_owner)) then
                assert(element_owner(sele)==ele)
             end if
          else if (no_found==2 .and. present(element_owner)) then
             ele=element_owner(sele)
             if (all(common_elements/=ele)) then
                FLAbort("Face not adjacent to specified element_owner")
             end if
             internal_face=.true.
          else
             FLAbort("Surface element in sndgln is not on the surface.")
          end if

          mesh%faces%face_element_list(sele)=ele

          ! now we have to fill in face_list:
          neigh=>row_m_ptr(mesh%faces%face_list, ele)
          faces=>row_ival_ptr(mesh%faces%face_list, ele)

          ! find the matching boundary of this element
          do j=1, facet_count(mesh%shape)
             if (neigh(j)<=0 .or. internal_face) then
                if (SetContains(snodes, mesh%ndglno( (ele-1)*nloc+ &
                     facet_dofs(mesh%shape, j)))) exit
             end if
          end do

          if (j>facet_count(mesh%shape)) then
             ! not found a matching facet, something's wrong
             FLAbort("Something wrong with the mesh, sndgln, or mesh%nelist")

          else if (neigh(j)/=0 .and. .not. internal_face) then
             ! this surface element is already registered
             ! so apparently there's a duplicate element in the surface mesh
             ewrite(0,*) 'Surface element:', faces(j),' and ',sele
             ewrite(0,*) 'Both define the surface element:', snodes
             FLAbort("Duplicate element in the surface mesh")
          end if

          ! register the surface element in face_list
          faces(j)=sele
          if (.not. internal_face) then
             neigh(j)=-j ! negative number indicates exterior facet
          else
             ! if all went well neigh(j) should have the right value:
             assert( neigh(j)==minval(common_elements, mask=common_elements/=ele) )
          end if

       end do

       bdry_count=stotel

    else

       bdry_count=0

    end if

    if (.not. (IsParallel() .or. present_and_true(incomplete_surface_mesh))) then
       ! register the rest of the boundaries
       ! remaining exterior boundaries first, thus completing the surface mesh
       ! This does not work in parallel and is therefore discouraged

       surface_elements_added=.false.
       do ele=1, size(mesh%faces%face_list,1)

          neigh=>row_m_ptr(mesh%faces%face_list, ele)
          faces=>row_ival_ptr(mesh%faces%face_list, ele)

          do j=1,size(neigh)

             if (neigh(j)==0) then

                bdry_count=bdry_count+1
                faces(j)=bdry_count
                neigh(j)=-j ! negative number indicates exterior facet

                surface_elements_added=.true.

             end if
          end do

       end do
       if (surface_elements_added .and. .not. warning_given) then
          ewrite(0,*) "WARNING: an incomplete surface mesh has been provided."
          ewrite(0,*) "This will not work in parallel."
          ewrite(0,*) "All parts of the domain boundary need to be marked with a (physical) surface id."
          warning_given=.true.
       end if
    end if

    ! the size of this array will be the way to store the n/o
    ! exterior boundaries (returned by surface_element_count())
    mesh%faces%boundary_ids_c = malloc(bdry_count * c_sizeof(1_c_int))
    call c_f_pointer(mesh%faces%boundary_ids_c, mesh%faces%boundary_ids,&
         & [bdry_count])
#ifdef HAVE_MEMORY_STATS
    call register_allocation("mesh_type", "integer", bdry_count, &
         trim(mesh%name)//" boundary_ids")
#endif

    mesh%faces%boundary_ids=0
    ! copy in supplied boundary ids
    if (present(boundary_ids)) then
       if (.not. present(sndgln)) then
          FLAbort("Boundary ids can only be supplied to add_faces with associated surface mesh")
       else if (stotel/=size(boundary_ids)) then
          FLAbort("Must supply boundary_ids array for the same number of elements as the surface mesh sndgln")
       end if
       mesh%faces%boundary_ids(1:stotel)=boundary_ids
    end if

    ! register the rest of the boundaries (the interior ones):
    do ele=1, size(mesh%faces%face_list,1)
       neigh=>row_m_ptr(mesh%faces%face_list, ele)
       faces=>row_ival_ptr(mesh%faces%face_list, ele)

       do j=1,size(neigh)
          if (neigh(j)>0 .and. faces(j)==0) then
             bdry_count=bdry_count+1
             faces(j)=bdry_count
          else if (neigh(j)==0) then
             ! left over exterior faces:
             bdry_count=bdry_count+1
             faces(j)=bdry_count
             neigh(j)=-j ! negative number indicates exterior boundary

          end if
          mesh%faces%local_face_number(faces(j))=j
       end do

       ! Record the element number of each face.
       ! (all faces should have an index now):
       mesh%faces%face_element_list(faces)=ele

    end do

    ! Sanity checks that we have found all the faces.
    assert(bdry_count==size(mesh%faces%face_list%sparsity%colm))
    assert(.not.any(mesh%faces%face_list%sparsity%colm==0))
    assert(.not.any(mesh%faces%face_list%ival==0))

  end subroutine add_faces_face_list

  subroutine add_faces_face_list_periodic_from_non_periodic_model( &
       mesh, model, periodic_face_map)
    ! computes the face_list of a periodic mesh by copying it from
    ! a non-periodic model mesh and changing the periodic faces
    ! to internal
    type(mesh_type), intent(inout):: mesh
    type(mesh_type), intent(in):: model
    type(integer_hash_table), intent(in):: periodic_face_map

    type(csr_sparsity):: face_list_sparsity
    integer, dimension(:), pointer :: faces, neigh
    integer:: face1, face2, ele1, ele2
    integer:: i, j

    ewrite(1,*) "In add_faces_face_list_periodic_from_non_periodic_model"

    ! for periodic meshes we need to fix the face_list
    ! so we have to have a separate copy
    call allocate(face_list_sparsity, element_count(model), &
         element_count(model), entries(model%faces%face_list), &
         name=trim(mesh%name)//"EEList")
    face_list_sparsity%colm=model%faces%face_list%sparsity%colm
    face_list_sparsity%findrm=model%faces%face_list%sparsity%findrm

    call allocate(mesh%faces%face_list, face_list_sparsity, &
         type=CSR_INTEGER, name=trim(mesh%name)//"FaceList")
    mesh%faces%face_list%ival=model%faces%face_list%ival
    call deallocate(face_list_sparsity)

    ! now fix the face list
    do i=1, key_count(periodic_face_map)
       call fetch_pair(periodic_face_map, i, face1, face2)
       ele1=model%faces%face_element_list(face1)
       ele2=model%faces%face_element_list(face2)

       ! register ele2 as a neighbour of ele1
       faces => row_ival_ptr(mesh%faces%face_list, ele1)
       neigh => row_m_ptr(mesh%faces%face_list, ele1)
       do j=1, size(faces)
          if (faces(j)==face1) then
             neigh(j)=ele2
          end if
       end do

       ! register ele1 as a neighbour of ele2
       faces => row_ival_ptr(mesh%faces%face_list, ele2)
       neigh => row_m_ptr(mesh%faces%face_list, ele2)
       do j=1, size(faces)
          if (faces(j)==face2) then
             neigh(j)=ele1
          end if
       end do

    end do

  end subroutine add_faces_face_list_periodic_from_non_periodic_model

  subroutine add_faces_face_list_non_periodic_from_periodic_model( &
       mesh, model, periodic_face_map, stat)
    ! computes the face_list of a non-periodic mesh by copying it from
    ! a periodic model mesh and changing the periodic faces
    ! to external
    type(mesh_type), intent(inout):: mesh
    type(mesh_type), intent(in):: model
    ! we return a list of periodic face pairs, as these need their local node numbering fixed later
    type(integer_hash_table), intent(out):: periodic_face_map
    integer, intent(out), optional :: stat

    type(csr_sparsity):: face_list_sparsity
    integer, dimension(:), pointer:: neigh, ele1_nodes, ele2_nodes
    integer:: face, face2, ele1, ele2, lface1, lface2

    ewrite(1,*) "In add_faces_face_list_non_periodic_from_periodic_model"

    if (present(stat)) then
       stat = 0
    end if

    ! we need to fix the face_list so we have to have a separate copy
    call allocate(face_list_sparsity, element_count(model), &
         element_count(model), entries(model%faces%face_list), &
         name=trim(mesh%name)//"EEList")
    face_list_sparsity%colm=model%faces%face_list%sparsity%colm
    face_list_sparsity%findrm=model%faces%face_list%sparsity%findrm

    call allocate(mesh%faces%face_list, face_list_sparsity, &
         type=CSR_INTEGER, name=trim(mesh%name)//"FaceList")
    mesh%faces%face_list%ival=model%faces%face_list%ival
    call deallocate(face_list_sparsity)

    call allocate(periodic_face_map)

    ! now fix the face list - by searching for internal faces in the
    ! model mesh, these are the periodic faces that now need to be removed
    do face = 1, surface_element_count(model)
       ele1 = face_ele(model, face)
       neigh => row_m_ptr(mesh%faces%face_list, ele1)
       lface1 = local_face_number(model, face)
       ele2 = neigh(lface1)
       if (ele2>0) then
          ! we've found an internal face in the surface mesh

          ! check that the connection has disappeared
          face2 = ele_face(model, ele2, ele1)
          lface2 = local_face_number(model, face2)
          ele1_nodes => ele_nodes(mesh, ele1)
          ele2_nodes => ele_nodes(mesh, ele2)
          if (SetContains( &
               ele1_nodes(facet_dofs(ele_shape(mesh, ele1), lface1)), &
               ele2_nodes(facet_dofs(ele_shape(mesh, ele2), lface2)))) then
             ! apparently these faces are still connected
             ! (not currently supported)
             if (present(stat)) then
                stat = 1
             else
                ewrite(-1,*) "Face: ", face, "; element: ", ele1
                ewrite(-1,*) "face_global_nodes(mesh, face): ", ele1_nodes(facet_dofs(ele_shape(mesh, ele1), lface1))
                ewrite(-1,*) "Opposing face: ", face2, "; element: ", ele2
                ewrite(-1,*) "face_global_nodes(mesh, face2): ", ele2_nodes(facet_dofs(ele_shape(mesh, ele2), lface2))
                FLAbort("Left-over internal faces in removing periodic bcs.")
             end if
          end if

          ! we're cool
          neigh(lface1)=-lface1
          ! might as well fix the other side while we're at it
          neigh => row_m_ptr(mesh%faces%face_list, ele2)
          neigh(lface2)=-lface2
          ! we store these as their face local node numbering needs to be "fixed" later
          call insert(periodic_face_map, face, face2)
       end if

    end do

  end subroutine add_faces_face_list_non_periodic_from_periodic_model

!!$  subroutine fix_periodic_face_orientation(nonperiodic, periodic, periodic_face_map)
!!$    !!< Fixes, i.e. overwrites the face local node numbering of non-periodic nonperiodic
!!$    !!< in periodic faces to make it consistent with the periodic 'mesh'
!!$    !!< Assumes the shape functions of elements and faces in mesh and nonperiodic are the same!!
!!$    type(mesh_type), intent(in):: nonperiodic
!!$    type(mesh_type), intent(in):: periodic
!!$    type(integer_hash_table), intent(in):: periodic_face_map
!!$
!!$    type(mesh_faces), pointer:: nonperiodic_faces
!!$    integer:: i, face1, face2
!!$
!!$    ewrite(1,*) "Inside fix_periodic_face_orientation"
!!$
!!$    if (.not. periodic%faces%shape==nonperiodic%faces%shape) then
!!$      ewrite(-1,*) "When deriving the faces structure of a periodic mesh from a non-periodic mesh"
!!$      ewrite(-1,*) "Its shape functions have to be the same"
!!$      FLAbort("Different shape functions in non-periodic nonperiodic mesh")
!!$    end if
!!$
!!$    do i=1, key_count(periodic_face_map)
!!$       call fetch_pair(periodic_face_map, i, face1, face2)
!!$       call fix_periodic_face_orientation_face(face1)
!!$       call fix_periodic_face_orientation_face(face2)
!!$    end do
!!$
!!$    ! when deriving a non-periodic mesh from a periodic model,
!!$    ! we don't have the surface mesh yet, so no need to fix it:
!!$    if (.not. associated(nonperiodic%faces%surface_node_list)) return
!!$
!!$    nonperiodic_faces => nonperiodic%faces
!!$    call deallocate(nonperiodic_faces%surface_mesh)
!!$#ifdef HAVE_MEMORY_STATS
!!$    call register_deallocation("mesh_type", "integer", &
!!$         size(nonperiodic_faces%surface_node_list), name='Surface'//trim(nonperiodic%name))
!!$#endif
!!$    deallocate(nonperiodic_faces%surface_node_list)
!!$    call create_surface_mesh(nonperiodic_faces%surface_mesh, &
!!$       nonperiodic_faces%surface_node_list, nonperiodic, name='Surface'//trim(nonperiodic%name))
!!$#ifdef HAVE_MEMORY_STATS
!!$    call register_allocation("mesh_type", "integer", &
!!$         size(nonperiodic_faces%surface_node_list), name='Surface'//trim(nonperiodic%name))
!!$#endif
!!$
!!$    contains
!!$
!!$    subroutine fix_periodic_face_orientation_face(face)
!!$    integer, intent(in):: face
!!$
!!$      integer, dimension(:), pointer:: mesh_face_local_nodes, nonperiodic_face_local_nodes
!!$
!!$      mesh_face_local_nodes => face_local_nodes(periodic, face)
!!$      nonperiodic_face_local_nodes => face_local_nodes(nonperiodic, face)
!!$
!!$      nonperiodic_face_local_nodes=mesh_face_local_nodes
!!$
!!$    end subroutine fix_periodic_face_orientation_face
!!$
!!$  end subroutine fix_periodic_face_orientation

  subroutine create_surface_mesh(surface_mesh, surface_nodes, &
       mesh, surface_elements, name)
    !! Creates a surface mesh consisting of the surface elements
    !! specified by surface_element_list
    type(mesh_type), intent(out):: surface_mesh
    !! Returns a pointer to a list containing the global node number
    !! of the nodes on this surface mesh, can be used for surface node
    !! to global node numbering conversion.
    integer, dimension(:), pointer:: surface_nodes
    !! mesh to take surface from (should have %faces component)
    type(mesh_type), intent(in):: mesh
    !! which surface elements to select
    !! (if not provided all surface elements are included)
    integer, dimension(:), optional, target,intent(in):: surface_elements
    !! name for the new surface_mesh
    character(len=*), intent(in):: name

    integer, dimension(:), pointer:: lsurface_elements
    integer, dimension(:), pointer:: suf_ndglno
    integer, dimension(:), allocatable:: nod2sufnod
    integer, dimension(mesh%faces%shape%ndof):: glnodes
    integer i, j, sele, sufnod, snloc
    type(mesh_type), pointer :: topology

    snloc=mesh%faces%shape%ndof

    if (present(surface_elements)) then
       lsurface_elements => surface_elements
    else
       allocate(lsurface_elements(1:surface_element_count(mesh)))
       lsurface_elements=(/ (i, i=1, size(lsurface_elements)) /)
    end if

    allocate(nod2sufnod(1:node_count(mesh)))
    nod2sufnod=0

    ! mark surface nodes with nod2sufnod(nod)==1
    sufnod=0
    do i=1, size(lsurface_elements)
       sele=lsurface_elements(i)
       glnodes=face_global_nodes(mesh, sele)
       do j=1, snloc
          if (nod2sufnod(glnodes(j))==0) then
             sufnod=sufnod+1
             nod2sufnod(glnodes(j))=1
          end if
       end do
    end do

    call allocate(surface_mesh, nodes=sufnod, &
         elements=size(lsurface_elements), shape=mesh%faces%shape, &
         name=name)

    surface_mesh%periodic=mesh%periodic
    surface_mesh%continuity=mesh%continuity

    allocate(surface_nodes(1:sufnod))

    ! create numbering in the same order as full nodal numbering:
    sufnod=0
    do i=1, size(nod2sufnod)
       if (nod2sufnod(i)==1) then
          sufnod=sufnod+1
          ! global node to surface node numbering
          nod2sufnod(i)=sufnod
          ! and the reverse
          surface_nodes(sufnod)=i

       end if
    end do

    ! map global node numbering to surface node numbering
    suf_ndglno => surface_mesh%ndglno
    do i=1, size(lsurface_elements)
       sele=lsurface_elements(i)
       suf_ndglno( (i-1)*snloc+1:i*snloc )=nod2sufnod(face_global_nodes(mesh, sele))
    end do

    deallocate(nod2sufnod)

    if (.not. present(surface_elements)) then
       deallocate(lsurface_elements)
    end if

    topology=>mesh_topology(mesh)
    surface_mesh%topology=topology%faces%surface_mesh

    if (associated(mesh%uid)) then
       surface_mesh%uid=>mesh%uid%mesh%faces%surface_mesh%uid
    end if

  end subroutine create_surface_mesh

  subroutine add_surface_uid(mesh)
    type(mesh_type), intent(inout) :: mesh
    type(mesh_type), pointer :: surface_mesh

    surface_mesh=>mesh%faces%surface_mesh

    assert(mesh%topology%refcount%id == mesh%refcount%id)
    allocate(surface_mesh%uid)
    ! This UID field is almost certainly leaked.
    call allocate(surface_mesh%uid, surface_mesh, trim(surface_mesh&
         &%name)//"CanonicalNumbering")
    surface_mesh%uid%val = mesh%uid%val(mesh%faces%surface_node_list)

  end subroutine add_surface_uid

  logical function SetContains(a, b)
    !!< Auxillary function that returns true if b contains a
    integer, dimension(:), intent(in):: a, b

    integer i

    SetContains=.false.
    do i=1, size(a)
       if (.not. any(b==a(i))) return
    end do
    SetContains=.true.

  end function SetContains

  function make_mesh_periodic(positions,physical_boundary_ids,aliased_boundary_ids,periodic_mapping_python,name, &
       periodic_face_map) result (positions_out)
    !!< Produce a mesh based on an old mesh but with periodic boundary conditions
    type(vector_field) :: positions_out
    type(vector_field), target, intent(in) :: positions

    integer, dimension(:), intent(in) :: physical_boundary_ids, aliased_boundary_ids
    character(len=*), intent(in) :: periodic_mapping_python
    ! name of positions_out%mesh, positions_out will called trim(name)//"Coordinate"
    character(len=*), intent(in), optional :: name
    !! builds up a map between aliased and physical faces, has to be allocated
    !! before the call, and is not emptied, so this can be used to build up a
    !! aliased to physical face map over multiple calls to make_mesh_periodic
    type(integer_hash_table), optional, intent(inout):: periodic_face_map

    type(mesh_type) :: mesh
    type(mesh_type), pointer:: model

    integer, dimension(:), allocatable :: ndglno
    real, dimension(:), pointer :: val
    integer, dimension(:,:), allocatable :: local_mapping_list
    integer, dimension(:), allocatable :: mapping_list, mapped
    integer :: i, j, k, id, nod, nod1, map_index, periodic_faces
    integer, dimension(:), allocatable :: face_nodes, face_nodes2
    integer :: count
    real, dimension(:,:), allocatable :: mapX
    real, dimension(positions%dim) :: tmp_pos
    real, dimension(:), pointer :: x,y,z
    real :: epsilon0
    logical :: found_node

    model => positions%mesh

    assert(has_faces(model))

    !get pointers to coordinates
    x => positions%val(1,:)
    if (positions%dim>1) then
       y => positions%val(2,:)
       if(positions%dim>1) then
          z => positions%val(3,:)
       end if
    end if
    call allocate(mesh, 0, model%elements, model%shape, name=name)
    !copy over all the mesh parameters
    mesh%continuity=model%continuity
    mesh%wrapped=.false.
    mesh%periodic=.true.

    if (associated(model%region_ids)) then
       call allocate_region_ids(mesh, size(model%region_ids))
       mesh%region_ids = model%region_ids
    end if

    !allocate memory for temporary place to hold old connectivity,
    !and memory for periodic connectivity
    allocate(ndglno(mesh%shape%cell%entity_counts(0)*model%elements), &
         mesh%ndglno(mesh%shape%ndof*model%elements))
#ifdef HAVE_MEMORY_STATS
    call register_allocation("mesh_type", "integer", size(mesh%ndglno), &
         name=mesh%name)
#endif

    !get old connectivity
    ndglno=model%ndglno

    !mapping_list is mapping from coordinates to periodic node number
    !mapped takes value 1 if node is aliased
    allocate( mapping_list(model%nodes), mapped(model%nodes) )
    mapping_list = 0
    mapped = 0

    !array to store the global node numbers in a face
    !ATTENTION: broken for meshes with different element types on
    !different meshes
    allocate( face_nodes(face_loc(model,1)) )
    allocate( face_nodes2(face_loc(model,1)) )

    !label any nodes which are aliased by:
    !   visiting each surface element
    !   checking the id
    !   if the id indicates aliased node then set mapped(node) to 1
    !           for each node in the surface element
    periodic_faces=0
    do i = 1, surface_element_count(model)
       id = surface_element_id(model,i)
       if(any(aliased_boundary_ids==id)) then
          face_nodes = face_global_nodes(model,i)
          mapped(face_nodes) = 1

          periodic_faces=periodic_faces+1
       end if
    end do

    !compute the number of nodes in the periodic mesh
    mesh%nodes = model%nodes - sum(mapped)

    ewrite(2,*) 'cjc nonods xnonod',mesh%nodes,model%nodes, sum(mapped)

    !local_mapping_list(1,:) contains aliased nodes
    !local_mapping_list(2,:) contains the nodes they are aliased to
    allocate( local_mapping_list(2,sum(mapped)) )
    local_mapping_list = 0

    !mapX contains the coordinates of the nodes that aliased nodes
    !are mapped to
    allocate( mapX(positions%dim,sum(mapped)) )

    !compute local_mapping_list(1,:) by
    !visiting each surface element
    !if id indicates it is aliased
    !visit each node in the surface element
    !if the node has yet to be added to the list (mapped(node)==1) then
    !     set mapped(node) to -1
    !     add node to the local_mapping_list(1,:), and
    !     increment count
    count = 1
    do i = 1, surface_element_count(model)
       id = surface_element_id(model,i)
       face_nodes = face_global_nodes(model,i)
       if(any(aliased_boundary_ids==id)) then
          do nod = 1, size(face_nodes)
             if(mapped(face_nodes(nod))==1) then
                mapped(face_nodes(nod))=-1
                local_mapping_list(1,count) = face_nodes(nod)
                count = count + 1
             end if
          end  do
       end if
    end do

    !compute the coordinates of the nodes that aliased nodes are mapped to
    if (positions%dim==1) then
       call set_from_python_function(mapX, &
            periodic_mapping_python, x(local_mapping_list(1,:)),  &
            time=0.0)
    else if (positions%dim==2) then
       call set_from_python_function(mapX, &
            periodic_mapping_python, x(local_mapping_list(1,:)),  &
            y(local_mapping_list(1,:)), time=0.0)
    else if (positions%dim==3) then
       call set_from_python_function(mapX, &
            periodic_mapping_python, x(local_mapping_list(1,:)),  &
            y(local_mapping_list(1,:)), z(local_mapping_list(1,:)), 0.0)
    end if

    !compute list of aliased to nodes
    !loop over surface elements
    !check for mapped to ids
    !if surface is mapped to
    !loop over nodes in aliased list
    !if any of the aliased nodes are mapped to a point
    !close to the current node
    !add that to the list
    do i = 1, surface_element_count(model)
       id = surface_element_id(model,i)
       if(any(physical_boundary_ids==id)) then
          face_nodes = face_global_nodes(model,i)
          do nod = 1, size(face_nodes)
             tmp_pos = node_val(positions, face_nodes(nod))
             found_node = .false.
             do nod1 = 1, size(mapX,2)
                !                epsilon0 = &
                !                     100*epsilon(0.0)*max( &
                !                              maxval(abs(my_vec)),maxval(abs(mapX(:,nod1))))
                epsilon0 = 1.0e-5
                if(maxval(abs(tmp_pos-mapX(:,nod1)))<epsilon0) then
                   local_mapping_list(2,nod1) = face_nodes(nod)
                   found_node = .true.
                end if
             end do
             if(.not.found_node) then
                do nod1 = 1, size(mapX,2)
                   ewrite(0,*) mapX(:,nod1)
                end do
                ewrite(0,*) 'node position is', tmp_pos
                FLExit('When deriving a periodic mesh, could not find node')
             end if
          end do
       end if
    end do

    ! now create a new periodic positions:
    call allocate(positions_out, positions%dim, mesh, name=trim(mesh%name)//"Coordinate")

    !check that it worked
    if(any(local_mapping_list==0)) then
       do nod = 1, size(local_mapping_list,2)
          ewrite(-1,*) local_mapping_list(1,nod), local_mapping_list(2,nod)
          ewrite(-1,*) node_val(positions, local_mapping_list(1,nod))
          ewrite(-1,*) node_val(positions, local_mapping_list(2,nod))
       end do
       FLExit('Failed to perform periodic mapping, check your periodic mappings.')
    end if

    !construct new numbering in mapping_list, first all the
    !aliased-to nodes
    mapping_list = 0
    do nod = 1, size(local_mapping_list,2)
       mapping_list(local_mapping_list(2,nod)) = nod
       mapping_list(local_mapping_list(1,nod)) = nod
       do k=1, positions_out%dim
          positions_out%val(k,nod)=node_val(positions, k, local_mapping_list(1,nod))
       end do
    end do
    !then the rest
    count = size(local_mapping_list,2)
    do nod = 1, size(mapping_list)
       if(mapping_list(nod)==0) then
          count = count + 1
          mapping_list(nod) = count
          do k=1, positions_out%dim
             positions_out%val(k,count)=node_val(positions, k, nod)
          end do
       end if
    end do

    !construct periodic ndglno
    mesh%ndglno = mapping_list(ndglno)

    !validate mesh
    if(maxval(mesh%ndglno) > mesh%nodes) then
       ewrite(-1,*) 'max(ndglno)=',maxval(mesh%ndglno), 'nodes',mesh%nodes
       FLAbort('Ndglno contains value greater than nonods')
    end if
    !    ! This test is quadratic in mesh%nodes
    !    do nod = 1, mesh%nodes
    !       if(.not.any(nod==mesh%ndglno)) then
    !          ewrite(-1,*) 'node',nod,'not present in ndglno'
    !          FLAbort('periodic mesh error!')
    !       end if
    !    end do

    if (has_faces(model) .and. present(periodic_face_map)) then

       map_index=0
       face_loop_1: do i=1,surface_element_count(model)
          if(any(physical_boundary_ids==surface_element_id(model,i))) then
             face_nodes = mapping_list(face_global_nodes(model,i))
             map_index=map_index+1

             do j=1,surface_element_count(model)
                if(any(aliased_boundary_ids==surface_element_id(model,j)))&
                     & then
                   face_nodes2 = mapping_list(face_global_nodes(model,j))

                   if (SetContains(face_nodes, face_nodes2)) then
                      call insert( periodic_face_map, i, j)

                      cycle face_loop_1
                   end if

                end if
             end do
             ! If we get here then we have an unmatched face.
             FLExit("Unmatched face in periodic mesh creation.  Check faces on periodic boundaries conform.")
          end if
       end do face_loop_1

    end if

    ! lose our reference
    call deallocate(mesh)

  end function make_mesh_periodic

  function make_submesh (model, name) &
       result (mesh)
    !!< Produce a mesh based on an old mesh but divided into piecewise linear.
    !!< FIXME: only works for quadratic simplex elements and doesn't do faces!
    type(mesh_type) :: mesh

    type(mesh_type), intent(in) :: model
    character(len=*), intent(in), optional :: name

    type(element_type) :: shape
    integer :: vertices, model_ele, sub_ele, l_ele
    integer, dimension(:,:), allocatable :: permutation
    integer, dimension(:), pointer :: model_nodes
    logical :: regions

    ewrite(1,*) 'entering make_submesh'

    if (present(name)) then
       mesh%name=name
    else
       mesh%name=empty_name
    end if

    allocate(mesh%adj_lists)
    mesh%continuity=model%continuity

    mesh%nodes = model%nodes

    vertices = vertex_count(model%shape)

    select case(cell_family(model%shape))
    case(FAMILY_SIMPLEX)

       select case(model%shape%degree)
       case(2)

          select case(vertices)
          case(3) ! triangle

             mesh%elements=4*model%elements

             allocate(permutation(4,3))
             ! here we assume that the one true node ordering is used
             permutation = reshape((/1, 2, 2, 4, &
                  2, 3, 4, 5, &
                  4, 5, 5, 6/), (/4,3/))
          case(4) ! tet

             mesh%elements=8*model%elements

             allocate(permutation(8,4))
             ! here we assume that the one true node ordering is used
             ! also we arbitrarily select a diagonal (between 5 and 7) through the central octahedron
             permutation = reshape((/1, 2, 4,  7, 2, 2, 4, 5, &
                  2, 3, 5,  8, 4, 5, 5, 7, &
                  4, 5, 6,  9, 5, 7, 7, 8, &
                  7, 8, 9, 10, 7, 8, 9, 9/), (/8,4/))
          case default
             ewrite(-1,*) "Submesh only supported for simplex elements."
             FLExit("Unsupported vertex count for the submesh.")
          end select
       case(1)
          !nothing to be done really

          mesh%elements=model%elements

          select case(vertices)
          case(3) ! triangle


             allocate(permutation(1,3))
             permutation = reshape((/1, 2, 3/), (/1,3/))
          case(4) ! tet

             allocate(permutation(1,4))
             permutation = reshape((/1, 2, 3, 4/), (/1,4/))
          case default
             ewrite(-1,*) "Submesh only supported for simplex elements."
             FLExit("Unsupported vertex count for the submesh.")
          end select

       case default
          FLExit("Submesh only supported for quadratic or lower elements")
       end select

    case default
       FLExit("Submesh only supported for simplex elements")
    end select

    shape = make_element_shape(vertices = vertices, dim = mesh_dim(model), &
         degree = 1, quad = model%shape%quadrature)

    mesh%shape=shape
    call incref(mesh%shape)

    regions = .false.
    if (associated(model%region_ids)) then
       call allocate_region_ids(mesh, size(model%region_ids))
       regions = .true.
    end if

    allocate(mesh%ndglno(mesh%shape%ndof*mesh%elements))
#ifdef HAVE_MEMORY_STATS
    call register_allocation("mesh_type", "integer", size(mesh%ndglno), &
         name=mesh%name)
#endif

    sub_ele = 0
    do model_ele = 1, element_count(model)
       model_nodes=>ele_nodes(model, model_ele)

       if(regions) mesh%region_ids(sub_ele+1:sub_ele+size(permutation,1)) = model%region_ids(model_ele)

       do l_ele = 1, size(permutation,1)
          sub_ele = sub_ele+1
          mesh%ndglno(mesh%shape%ndof*(sub_ele-1)+1:mesh%shape%ndof*sub_ele) = model_nodes(permutation(l_ele,:))
       end do

    end do

    mesh%wrapped=.false.
    mesh%periodic=model%periodic
    nullify(mesh%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    call addref(mesh)
    call deallocate(shape)

  end function make_submesh

  function extract_elements(positions, elements) result(subpos)
    !! Given a mesh and a list of elements,
    !! return a mesh containing just those elements.
    type(vector_field), intent(in), target :: positions
    integer, dimension(:), intent(in) :: elements
    type(vector_field) :: subpos

    type(mesh_type), pointer :: mesh
    type(mesh_type) :: submesh

    integer :: ele, nodes, i, j, k, loc

    mesh => positions%mesh
    loc = ele_loc(mesh, 1)

    nodes = size(elements) * loc
    call allocate(submesh, nodes, size(elements), ele_shape(mesh, 1), "SubMesh")
    call allocate(subpos, positions%dim, submesh, "SubCoordinate")
    call deallocate(submesh)

    do j=1,nodes
       submesh%ndglno(j) = j
    end do

    j = 1
    do i=1,size(elements)
       ele = elements(i)
       do k=1,positions%dim
          subpos%val(k,ele_nodes(subpos, j)) = ele_val(positions, k, ele)
       end do
       j = j + 1
    end do
  end function extract_elements

  subroutine add_lists_mesh(mesh, nnlist, nelist, eelist)
    !!< Add requested adjacency lists to the adjacency cache for the supplied mesh

    type(mesh_type), intent(in) :: mesh
    logical, optional, intent(in) :: nnlist, nelist, eelist

    type(csr_sparsity), pointer :: lnnlist, lnelist, leelist

    logical :: ladd_nnlist, ladd_nelist, ladd_eelist

    assert(associated(mesh%adj_lists))
    ladd_nnlist = present_and_true(nnlist) .and. .not. associated(mesh%adj_lists%nnlist)
    ladd_eelist = present_and_true(eelist) .and. .not. associated(mesh%adj_lists%eelist)
    ladd_nelist = (present_and_true(nelist) .or. ladd_eelist) .and. .not. associated(mesh%adj_lists%nelist)

    if(ladd_nnlist .and. ladd_nelist .and. ladd_eelist) then
       ewrite(2, *) "Adding node-node list to mesh " // trim(mesh%name)
       ewrite(2, *) "Adding node-element list to mesh " // trim(mesh%name)
       ewrite(2, *) "Adding element-element list to mesh " // trim(mesh%name)
       allocate(mesh%adj_lists%nnlist)
       allocate(mesh%adj_lists%nelist)
       allocate(mesh%adj_lists%eelist)
       ! Use these pointers to work around compilers that insist on having mesh
       ! intent(inout) - it really only needs to be intent(in)
       lnnlist => mesh%adj_lists%nnlist
       lnelist => mesh%adj_lists%nelist
       leelist => mesh%adj_lists%eelist
       call makelists(mesh, &
            & nnlist = lnnlist, &
            & nelist = lnelist, &
            & eelist = leelist)
    else if(ladd_nnlist .and. ladd_nelist) then
       ewrite(2, *) "Adding node-node list to mesh " // trim(mesh%name)
       ewrite(2, *) "Adding node-element list to mesh " // trim(mesh%name)
       allocate(mesh%adj_lists%nnlist)
       allocate(mesh%adj_lists%nelist)
       ! Use these pointers to work around compilers that insist on having mesh
       ! intent(inout) - it really only needs to be intent(in)
       lnnlist => mesh%adj_lists%nnlist
       lnelist => mesh%adj_lists%nelist
       call makelists(mesh, &
            & nnlist = lnnlist, &
            & nelist = lnelist)
    else if(ladd_eelist) then
       if(ladd_nnlist) then
          call add_nnlist(mesh)
       end if
       call add_eelist(mesh)  ! The eelist generates the nelist. If we need all
       ! three then we enter the branch above.
    else if(ladd_nnlist) then
       call add_nnlist(mesh)
    else if(ladd_nelist) then
       call add_nelist(mesh)
       !    else
       !      ! We already have the requested lists (or no lists were requested)
    end if

  end subroutine add_lists_mesh

  subroutine add_lists_scalar(field, nnlist, nelist, eelist)
    !!< Add requested adjacency lists to the adjacency cache for the supplied field

    type(scalar_field), intent(in) :: field
    logical, optional, intent(in) :: nnlist, nelist, eelist

    call add_lists(field%mesh, nnlist = nnlist, nelist = nelist, eelist = eelist)

  end subroutine add_lists_scalar

  subroutine add_lists_vector(field, nnlist, nelist, eelist)
    !!< Add requested adjacency lists to the adjacency cache for the supplied field

    type(vector_field), intent(in) :: field
    logical, optional, intent(in) :: nnlist, nelist, eelist

    call add_lists(field%mesh, nnlist = nnlist, nelist = nelist, eelist = eelist)

  end subroutine add_lists_vector

  subroutine add_lists_tensor(field, nnlist, nelist, eelist)
    !!< Add requested adjacency lists to the adjacency cache for the supplied field

    type(tensor_field), intent(in) :: field
    logical, optional, intent(in) :: nnlist, nelist, eelist

    call add_lists(field%mesh, nnlist = nnlist, nelist = nelist, eelist = eelist)

  end subroutine add_lists_tensor

  subroutine extract_lists_mesh(mesh, nnlist, nelist, eelist)
    !!< Extract adjacancy lists (generating if necessary) from the
    !!< adjacency cache for the supplied mesh

    type(mesh_type), intent(in) :: mesh
    type(csr_sparsity), optional, intent(out) :: nnlist
    type(csr_sparsity), optional, intent(out) :: nelist
    type(csr_sparsity), optional, intent(out) :: eelist

    call add_lists(mesh, nnlist = present(nnlist), nelist = present(nelist), eelist = present(eelist))
    assert(associated(mesh%adj_lists))
    if(present(nnlist)) then
       assert(associated(mesh%adj_lists%nnlist))
       nnlist = mesh%adj_lists%nnlist
       assert(has_references(nnlist))
    end if
    if(present(nelist)) then
       assert(associated(mesh%adj_lists%nelist))
       nelist = mesh%adj_lists%nelist
       assert(has_references(nelist))
    end if
    if(present(eelist)) then
       assert(associated(mesh%adj_lists%eelist))
       eelist = mesh%adj_lists%eelist
       assert(has_references(eelist))
    end if

  end subroutine extract_lists_mesh

  subroutine extract_lists_scalar(field, nnlist, nelist, eelist)
    !!< Extract adjacancy lists (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(scalar_field), intent(in) :: field
    type(csr_sparsity), optional, intent(out) :: nnlist
    type(csr_sparsity), optional, intent(out) :: nelist
    type(csr_sparsity), optional, intent(out) :: eelist

    call extract_lists(field%mesh, nnlist = nnlist, nelist = nelist, eelist = eelist)

  end subroutine extract_lists_scalar

  subroutine extract_lists_vector(field, nnlist, nelist, eelist)
    !!< Extract adjacancy lists (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(vector_field), intent(in) :: field
    type(csr_sparsity), optional, intent(out) :: nnlist
    type(csr_sparsity), optional, intent(out) :: nelist
    type(csr_sparsity), optional, intent(out) :: eelist

    call extract_lists(field%mesh, nnlist = nnlist, nelist = nelist, eelist = eelist)

  end subroutine extract_lists_vector

  subroutine extract_lists_tensor(field, nnlist, nelist, eelist)
    !!< Extract adjacancy lists (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(tensor_field), intent(in) :: field
    type(csr_sparsity), optional, intent(out) :: nnlist
    type(csr_sparsity), optional, intent(out) :: nelist
    type(csr_sparsity), optional, intent(out) :: eelist

    call extract_lists(field%mesh, nnlist = nnlist, nelist = nelist, eelist = eelist)

  end subroutine extract_lists_tensor

  subroutine add_nnlist_mesh(mesh)
    !!< Add the node-node list to the adjacency cache for the supplied mesh

    type(mesh_type), intent(in) :: mesh

    type(csr_sparsity), pointer :: nnlist

    assert(associated(mesh%adj_lists))
    if(.not. associated(mesh%adj_lists%nnlist)) then
       ewrite(2, *) "Adding node-node list to mesh " // trim(mesh%name)
       allocate(mesh%adj_lists%nnlist)
       ! Use this pointer to work around compilers that insist on having mesh
       ! intent(inout) - it really only needs to be intent(in)
       nnlist => mesh%adj_lists%nnlist
       call makelists(mesh, nnlist = nnlist)
#ifdef DDEBUG
    else
       assert(has_references(mesh%adj_lists%nnlist))
#endif
    end if

  end subroutine add_nnlist_mesh

  subroutine add_nnlist_scalar(field)
    !!< Add the node-node list to the adjacency cache for the supplied field

    type(scalar_field), intent(in) :: field

    call add_nnlist(field%mesh)

  end subroutine add_nnlist_scalar

  subroutine add_nnlist_vector(field)
    !!< Add the node-node list to the adjacency cache for the supplied field

    type(vector_field), intent(in) :: field

    call add_nnlist(field%mesh)

  end subroutine add_nnlist_vector

  subroutine add_nnlist_tensor(field)
    !!< Add the node-node list to the adjacency cache for the supplied field

    type(tensor_field), intent(in) :: field

    call add_nnlist(field%mesh)

  end subroutine add_nnlist_tensor

  function extract_nnlist_mesh(mesh) result(nnlist)
    !!< Extract the node-node list (generating if necessary) from the
    !!< adjacency cache for the supplied mesh

    type(mesh_type), intent(in) :: mesh

    type(csr_sparsity), pointer :: nnlist

    call add_nnlist(mesh)
    nnlist => mesh%adj_lists%nnlist
    assert(has_references(nnlist))

  end function extract_nnlist_mesh

  function extract_nnlist_scalar(field) result(nnlist)
    !!< Extract the node-node list (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(scalar_field), intent(in) :: field

    type(csr_sparsity), pointer :: nnlist

    nnlist => extract_nnlist(field%mesh)

  end function extract_nnlist_scalar

  function extract_nnlist_vector(field) result(nnlist)
    !!< Extract the node-node list (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(vector_field), intent(in) :: field

    type(csr_sparsity), pointer :: nnlist

    nnlist => extract_nnlist(field%mesh)

  end function extract_nnlist_vector

  function extract_nnlist_tensor(field) result(nnlist)
    !!< Extract the node-node list (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(tensor_field), intent(in) :: field

    type(csr_sparsity), pointer :: nnlist

    nnlist => extract_nnlist(field%mesh)

  end function extract_nnlist_tensor

  subroutine add_nelist_mesh(mesh)
    !!< Add the node-element list to the adjacency cache for the supplied mesh

    type(mesh_type), intent(in) :: mesh

    type(csr_sparsity), pointer :: nelist

    assert(associated(mesh%adj_lists))
    if(.not. associated(mesh%adj_lists%nelist)) then
       ewrite(2, *) "Adding node-element list to mesh " // trim(mesh%name)
       allocate(mesh%adj_lists%nelist)
       ! Use this pointer to work around compilers that insist on having mesh
       ! intent(inout) - it really only needs to be intent(in)
       nelist => mesh%adj_lists%nelist
       call makelists(mesh, nelist = nelist)
#ifdef DDEBUG
    else
       assert(has_references(mesh%adj_lists%nelist))
#endif
    end if

  end subroutine add_nelist_mesh

  subroutine add_nelist_scalar(field)
    !!< Add the node-element list to the adjacency cache for the supplied field

    type(scalar_field), intent(in) :: field

    call add_nelist(field%mesh)

  end subroutine add_nelist_scalar

  subroutine add_nelist_vector(field)
    !!< Add the node-element list to the adjacency cache for the supplied field

    type(vector_field), intent(in) :: field

    call add_nelist(field%mesh)

  end subroutine add_nelist_vector

  subroutine add_nelist_tensor(field)
    !!< Add the node-element list to the adjacency cache for the supplied field

    type(tensor_field), intent(in) :: field

    call add_nelist(field%mesh)

  end subroutine add_nelist_tensor

  function extract_nelist_mesh(mesh) result(nelist)
    !!< Extract the node-element list (generating if necessary) from the
    !!< adjacency cache for the supplied mesh

    type(mesh_type), intent(in) :: mesh

    type(csr_sparsity), pointer :: nelist

    call add_nelist(mesh)
    nelist => mesh%adj_lists%nelist
    assert(has_references(nelist))

  end function extract_nelist_mesh

  function extract_nelist_scalar(field) result(nelist)
    !!< Extract the node-element list (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(scalar_field), intent(in) :: field

    type(csr_sparsity), pointer :: nelist

    nelist => extract_nelist(field%mesh)

  end function extract_nelist_scalar

  function extract_nelist_vector(field) result(nelist)
    !!< Extract the node-element list (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(vector_field), intent(in) :: field

    type(csr_sparsity), pointer :: nelist

    nelist => extract_nelist(field%mesh)

  end function extract_nelist_vector

  function extract_nelist_tensor(field) result(nelist)
    !!< Extract the node-element list (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(tensor_field), intent(in) :: field

    type(csr_sparsity), pointer :: nelist

    nelist => extract_nelist(field%mesh)

  end function extract_nelist_tensor

  subroutine add_eelist_mesh(mesh)
    !!< Add the element-element list to the adjacency cache for the supplied mesh

    type(mesh_type), intent(in) :: mesh

    type(csr_sparsity), pointer :: eelist, nelist

    assert(associated(mesh%adj_lists))
    if(.not. associated(mesh%adj_lists%eelist)) then
       ewrite(2, *) "Adding element-element list to mesh " // trim(mesh%name)
       allocate(mesh%adj_lists%eelist)
       ! Use this pointer to work around compilers that insist on having mesh
       ! intent(inout) - it really only needs to be intent(in)
       eelist => mesh%adj_lists%eelist
       ! We need the nelist to generate the eelist, so extract it from the cache
       ! (generating if necessary)
       nelist => extract_nelist(mesh)
       ewrite(1, *) "Using the new makeeelist"
       call makeeelist(eelist, mesh, nelist)
#ifdef DDEBUG
    else
       assert(has_references(mesh%adj_lists%eelist))
#endif
    end if

  end subroutine add_eelist_mesh

  subroutine add_eelist_scalar(field)
    !!< Add the element-element list to the adjacency cache for the supplied field

    type(scalar_field), intent(in) :: field

    call add_eelist(field%mesh)

  end subroutine add_eelist_scalar

  subroutine add_eelist_vector(field)
    !!< Add the element-element list to the adjacency cache for the supplied field

    type(vector_field), intent(in) :: field

    call add_eelist(field%mesh)

  end subroutine add_eelist_vector

  subroutine add_eelist_tensor(field)
    !!< Add the element-element list to the adjacency cache for the supplied field

    type(tensor_field), intent(in) :: field

    call add_eelist(field%mesh)

  end subroutine add_eelist_tensor

  function extract_eelist_mesh(mesh) result(eelist)
    !!< Extract the element-element list (generating if necessary) from the
    !!< adjacency cache for the supplied mesh

    type(mesh_type), intent(in) :: mesh

    type(csr_sparsity), pointer :: eelist

    call add_eelist(mesh)
    eelist => mesh%adj_lists%eelist
    assert(has_references(eelist))

  end function extract_eelist_mesh

  function extract_eelist_scalar(field) result(eelist)
    !!< Extract the element-element list (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(scalar_field), intent(in) :: field

    type(csr_sparsity), pointer :: eelist

    eelist => extract_eelist(field%mesh)

  end function extract_eelist_scalar

  function extract_eelist_vector(field) result(eelist)
    !!< Extract the element-element list (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(vector_field), intent(in) :: field

    type(csr_sparsity), pointer :: eelist

    eelist => extract_eelist(field%mesh)

  end function extract_eelist_vector

  function extract_eelist_tensor(field) result(eelist)
    !!< Extract the element-element list (generating if necessary) from the
    !!< adjacency cache for the supplied field

    type(tensor_field), intent(in) :: field

    type(csr_sparsity), pointer :: eelist

    eelist => extract_eelist(field%mesh)

  end function extract_eelist_tensor

  subroutine remove_lists_mesh(mesh)
    !!< Remove the adjecency lists from the adjacency cache for the supplied
    !!< mesh

    type(mesh_type), intent(inout) :: mesh

    call remove_nnlist(mesh)
    call remove_nelist(mesh)
    call remove_eelist(mesh)

  end subroutine remove_lists_mesh

  subroutine remove_nnlist_mesh(mesh)
    !!< Remove the node-node list from the adjacency cache for the supplied mesh

    type(mesh_type), intent(inout) :: mesh

    assert(associated(mesh%adj_lists))
    if(associated(mesh%adj_lists%nnlist)) then
       ewrite(2, *) "Removing node-node list from mesh " // trim(mesh%name)
       call deallocate(mesh%adj_lists%nnlist)
       deallocate(mesh%adj_lists%nnlist)
       nullify(mesh%adj_lists%nnlist)
    end if

  end subroutine remove_nnlist_mesh

  subroutine remove_nelist_mesh(mesh)
    !!< Remove the node-element list from the adjacency cache for the supplied mesh

    type(mesh_type), intent(inout) :: mesh

    assert(associated(mesh%adj_lists))
    if(associated(mesh%adj_lists%nelist)) then
       ewrite(2, *) "Removing node-element list from mesh " // trim(mesh%name)
       call deallocate(mesh%adj_lists%nelist)
       deallocate(mesh%adj_lists%nelist)
       nullify(mesh%adj_lists%nelist)
    end if

  end subroutine remove_nelist_mesh

  subroutine remove_eelist_mesh(mesh)
    !!< Remove the element-element list from the adjacency cache for the supplied mesh

    type(mesh_type), intent(inout) :: mesh

    assert(associated(mesh%adj_lists))
    if(associated(mesh%adj_lists%eelist)) then
       ewrite(2, *) "Removing element-element list from mesh " // trim(mesh%name)
       call deallocate(mesh%adj_lists%eelist)
       deallocate(mesh%adj_lists%eelist)
       nullify(mesh%adj_lists%eelist)
    end if

  end subroutine remove_eelist_mesh


  subroutine zero_scalar(field)
    !!< Set all entries in the field provided to 0.0
    type(scalar_field), intent(inout) :: field
#ifdef _OPENMP
    integer :: i
#endif

    assert(field%field_type/=FIELD_TYPE_PYTHON)

#ifdef _OPENMP
    ! Use first touch policy.
    !$OMP PARALLEL DO SCHEDULE(STATIC)
    do i=1, size(field%val)
       field%val(i)=0.0
    end do
    !$OMP END PARALLEL DO
#else
    field%val=0.0
#endif

  end subroutine zero_scalar

  subroutine zero_vector(field)
    !!< Set all entries in the field provided to 0.0
    type(vector_field), intent(inout) :: field

#ifdef _OPENMP
    integer :: i
#endif

    assert(field%field_type/=FIELD_TYPE_PYTHON)

#ifdef _OPENMP
    ! Use first touch policy.
    !$OMP PARALLEL DO SCHEDULE(STATIC)
    do i=1, size(field%val, 2)
       field%val(:,i)=0.0
    end do
    !$OMP END PARALLEL DO
#else
    field%val=0.0
#endif

  end subroutine zero_vector

  subroutine zero_vector_dim(field, dim)
    !!< Set all entries in dimension dim of the field provided to 0.0
    type(vector_field), intent(inout) :: field
    integer, intent(in) :: dim

#ifdef _OPENMP
    integer :: j
#endif

    assert(field%field_type/=FIELD_TYPE_PYTHON)

#ifdef _OPENMP
    ! Use first touch policy.
    !$OMP PARALLEL DO SCHEDULE(STATIC)
    do j=1, size(field%val, 2)
       field%val(dim,j)=0.0
    end do
    !$OMP END PARALLEL DO
#else
    field%val(dim,:)=0.0
#endif

  end subroutine zero_vector_dim

  subroutine zero_tensor(field)
    !!< Set all entries in the field provided to 0.0
    type(tensor_field), intent(inout) :: field

#ifdef _OPENMP
    integer :: j
#endif

    assert(field%field_type/=FIELD_TYPE_PYTHON)

#ifdef _OPENMP
    ! Use first touch policy.
    !$OMP PARALLEL DO SCHEDULE(STATIC)
    do j=1, size(field%val, 3)
       field%val(:,:,j)=0.0
    end do
    !$OMP END PARALLEL DO
#else
    field%val=0.0
#endif

  end subroutine zero_tensor

  subroutine zero_tensor_dim_dim(field, dim1, dim2)
    !!< Set all entries in the component indicated of field to 0.0
    type(tensor_field), intent(inout) :: field
    integer, intent(in) :: dim1, dim2

#ifdef _OPENMP
    integer :: j
#endif

    assert(field%field_type/=FIELD_TYPE_PYTHON)

#ifdef _OPENMP
    ! Use first touch policy.
    !$OMP PARALLEL DO SCHEDULE(STATIC)
    do j=1, size(field%val, 3)
       field%val(dim1,dim2,j)=0.0
    end do
    !$OMP END PARALLEL DO
#else
    field%val(dim1,dim2,:)=0.0
#endif

  end subroutine zero_tensor_dim_dim

  subroutine zero_scalar_field_nodes(field, node_numbers)
    !!< Zeroes the scalar field at the specified node_numbers
    !!< Does not work for constant fields
    type(scalar_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers

    assert(field%field_type==FIELD_TYPE_NORMAL)

    field%val(node_numbers) = 0.0

  end subroutine zero_scalar_field_nodes

  subroutine zero_vector_field_nodes(field, node_numbers)
    !!< Zeroes the vector field at the specified nodes
    !!< Does not work for constant fields
    type(vector_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers
    integer :: i

    assert(field%field_type==FIELD_TYPE_NORMAL)

    do i=1,field%dim
       field%val(i,node_numbers) = 0.0
    end do

  end subroutine zero_vector_field_nodes

  subroutine zero_tensor_field_nodes(field, node_numbers)
    !!< Zeroes the tensor field at the specified nodes
    !!< Does not work for constant fields
    type(tensor_field), intent(inout) :: field
    integer, dimension(:), intent(in) :: node_numbers

    assert(field%field_type==FIELD_TYPE_NORMAL)

    field%val(:, :, node_numbers) = 0.0

  end subroutine zero_tensor_field_nodes

  subroutine allocate_region_ids(mesh, size)
    type(mesh_type), intent(inout) :: mesh
    integer, intent(in) :: size

    mesh%region_ids_c = malloc(size*c_sizeof(1_c_int))
    call c_f_pointer(mesh%region_ids_c, mesh%region_ids, [size])

  end subroutine allocate_region_ids

  subroutine deallocate_region_ids(mesh)
    type(mesh_type), intent(inout) :: mesh

    call free(mesh%region_ids_c)
    mesh%region_ids_c=C_NULL_PTR
    mesh%region_ids=>NULL()

  end subroutine deallocate_region_ids

#include "Reference_count_mesh_type.F90"
#include "Reference_count_scalar_field.F90"
#include "Reference_count_vector_field.F90"
#include "Reference_count_tensor_field.F90"

end module fields_allocates
