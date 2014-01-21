!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    David Ham
!    Department of Computing
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
module python_interface_f
  ! This module provides a C interface to some Fluidity functions to enable
  !  these to be called from Python.
  use iso_c_binding
  use elements
  use global_parameters, only: malloc, free, c_sizeof
  use fields_data_types
  use halo_data_types
  use fields_allocates
  use fields_base
  use fields_halos
  use parallel_tools
  use halos_registration
  use parallel_fields
  use mesh_files

  implicit none

  type, bind(c) :: element_t
     !! Type containing the essential information from a FIAT element to
     !!  produce a Fluidity element.
     integer(c_int) :: dimension
     integer(c_int) :: vertices
     integer(c_int) :: ndof
     integer(c_int) :: degree
     type(c_ptr) :: dofs_per
     type(c_ptr) :: entity_dofs
  end type element_t

  type, bind(c) :: mesh_t
     !! Type containing the essential information from a fluidity mesh
     !!  required for the construction of a Python Mesh.
     type(c_ptr) :: element_vertex_list
     integer(c_int) :: dimension
     integer(c_int) :: cell_vertices ! Number of vertices per cell.
     integer(c_int) :: vertex_count
     integer(c_int) :: cell_count
     integer(c_int) :: exterior_facet_count
     integer(c_int) :: interior_facet_count
     type(c_ptr) :: cell_classes
     type(c_ptr) :: vertex_classes
     type(c_ptr) :: region_ids
     integer(c_int) :: uid
     integer(c_int) :: space_dimension
     type(c_ptr) :: coordinates
     type(c_ptr) :: fluidity_coordinate
     type(c_ptr) :: fluidity_mesh
     type(c_ptr) :: interior_local_facet_number
     type(c_ptr) :: exterior_local_facet_number
     type(c_ptr) :: interior_facet_cell
     type(c_ptr) :: exterior_facet_cell
     type(c_ptr) :: boundary_ids
  end type mesh_t

  enum, bind(c)
     enumerator :: VERTEX, CELL
  end enum

  type, bind(c) :: halo_t
     !! Type containing information in Fluidity to construct a Python
     !! Halo object.
     type(c_ptr) :: sends
     type(c_ptr) :: nsends
     type(c_ptr) :: receives
     type(c_ptr) :: nreceives
     integer(c_int) :: nowned_nodes
     integer(c_int) :: nprocs
     integer(c_int) :: entity_type
     type(c_ptr) :: receives_global_to_universal
     integer(c_int) :: receives_global_to_universal_len
     integer(c_int) :: universal_offset
     integer(c_int) :: comm
     type(c_ptr) :: fluidity_halo
  end type halo_t

  type, bind(c) :: function_space_t
     !! Type containing the essential information from a fluidity mesh
     !! required for the construction of a Python FunctionSpace
     type(c_ptr) :: element_dof_list
     type(c_ptr) :: fluidity_mesh
     integer(c_int) :: element_count
     integer(c_int) :: dof_count
     type(c_ptr) :: dof_classes
     type(c_ptr) :: interior_facet_dof_list
     type(c_ptr) :: exterior_facet_dof_list
  end type function_space_t

  private
  public element_t, function_space_f, read_mesh_f
  public function_space_destructor_f
  public vector_field_destructor_f

contains

  function function_space_f(mesh_ptr, fiat_element) bind (c)
    type(c_ptr), value, intent(in) :: mesh_ptr
    type(element_t) :: fiat_element
    type(function_space_t) :: function_space_f

    type(mesh_type), pointer :: mesh, fluidity_function_space
    type(element_type) :: element

    call c_f_pointer(mesh_ptr, mesh)

    allocate(fluidity_function_space)

    element = make_element_shape_from_fiat(fiat_element)

    fluidity_function_space = make_mesh(mesh, element, with_faces=.false.)

    call populate_facet_maps(fluidity_function_space)

    function_space_f%fluidity_mesh = c_loc(fluidity_function_space)

    function_space_f%element_dof_list = fluidity_function_space%ndglno_c

    function_space_f%element_count = fluidity_function_space%elements

    function_space_f%dof_count = fluidity_function_space%nodes

    function_space_f%dof_classes = c_loc(fluidity_function_space%node_classes)

    function_space_f%interior_facet_dof_list = &
         fluidity_function_space%interior_facet_dof_list

    function_space_f%exterior_facet_dof_list = &
         fluidity_function_space%exterior_facet_dof_list

  contains

    subroutine populate_facet_maps(fs)
      ! Populate the firedrake facet maps for interior and exterior facets.
      type(mesh_type), intent(inout) :: fs

      integer, pointer :: map(:,:), facet_cells(:,:), facet_cell(:)

      integer :: dofs, facets, ifacets, efacets, f

      facets = face_count(fs%topology)
      efacets = surface_element_count(fs%topology)
      ifacets = (facets - efacets)/2
      assert(2*ifacets == facets - efacets)
      dofs = ele_loc(fs,1)

      ! Populate interior facet map.
      fs%interior_facet_dof_list = malloc(2*dofs*ifacets*c_sizeof(1_c_int))

      call c_f_pointer(fs%interior_facet_dof_list, map, [2*dofs,ifacets])

      call c_f_pointer(fs%topology%faces%interior_facet_cell, &
           facet_cells, [2, ifacets])

      do f = 1, ifacets
         ! +1 and -1 to change numbering systems between c and Fortran
         map(1:dofs, f) = ele_nodes(fs, facet_cells(1, f)+1) - 1
         map(dofs+1:, f) = ele_nodes(fs, facet_cells(2, f)+1) - 1
      end do

      ! Populate exterior facet map.
      fs%exterior_facet_dof_list = malloc(2*dofs*efacets*c_sizeof(1_c_int))

      call c_f_pointer(fs%exterior_facet_dof_list, map, [dofs,efacets])

      call c_f_pointer(fs%topology%faces%exterior_facet_cell, &
           facet_cell, [efacets])

      do f = 1, efacets
         map(:, f) = ele_nodes(fs, facet_cell(f) + 1) - 1
      end do

    end subroutine populate_facet_maps

  end function function_space_f

  function extruded_mesh_f(mesh_ptr, fiat_element, dofs_per_column_ptr) bind (c)
    type(c_ptr), value, intent(in) :: mesh_ptr
    type(element_t) :: fiat_element
    type(function_space_t) :: extruded_mesh_f

    type(mesh_type), pointer :: mesh, fluidity_function_space
    type(element_type) :: element
    type(c_ptr), intent(in), value :: dofs_per_column_ptr
    integer, dimension(:), pointer:: dofs_per_column

    call c_f_pointer(mesh_ptr, mesh)

    allocate(fluidity_function_space)

    element = make_element_shape_from_fiat(fiat_element)

    call c_f_pointer(dofs_per_column_ptr, dofs_per_column, [element%dim+1])

    fluidity_function_space = make_mesh(mesh, element, dofs_per_column=dofs_per_column, with_faces=.false.)

    extruded_mesh_f%fluidity_mesh = c_loc(fluidity_function_space)

    extruded_mesh_f%element_dof_list = fluidity_function_space%ndglno_c

    extruded_mesh_f%element_count = fluidity_function_space%elements

    extruded_mesh_f%dof_count = fluidity_function_space%nodes

    extruded_mesh_f%dof_classes = c_loc(fluidity_function_space%node_classes)

  end function extruded_mesh_f


  function halo_f(mesh_ptr, halo_entity_type) bind(c)
    type(c_ptr), value, intent(in) :: mesh_ptr

    integer(c_int), value, intent(in) :: halo_entity_type
    type(halo_t) :: halo_f
    type(mesh_type), pointer :: mesh
    type(halo_type), pointer :: halo => null()
    integer :: i
    integer, dimension(:), pointer :: tmp

    call c_f_pointer(mesh_ptr, mesh)

    select case (halo_entity_type)
    case (VERTEX)
       if (associated(mesh%halos)) then
          halo => mesh%halos(halo_count(mesh))
       end if
    case (CELL)
       if (associated(mesh%element_halos)) then
          halo => mesh%element_halos(halo_count(mesh))
       end if
    case default
       FLAbort("Unknown halo entity type")
    end select

    if (.not.associated(halo)) then
       halo_f%nprocs = -1
       return
    end if

    halo_f%nprocs = halo%nprocs
    halo_f%nowned_nodes = halo%nowned_nodes
    halo_f%universal_offset = halo%my_owned_nodes_unn_base
    halo_f%receives_global_to_universal = halo%receives_gnn_to_unn_c
    halo_f%receives_global_to_universal_len = size(halo%receives_gnn_to_unn)
    halo_f%sends = halo%sends_c

    ! Gets freed when building the python object.
    halo_f%nsends = malloc(size(halo%sends) * c_sizeof(1_c_int))
    call c_f_pointer(halo_f%nsends, tmp, [size(halo%sends)])
    do i = 1, size(halo%sends)
       tmp(i) = size(halo%sends(i)%ptr)
    end do

    halo_f%receives = halo%receives_c

    ! Gets freed when building the python object.
    halo_f%nreceives = malloc(size(halo%receives) * c_sizeof(1_c_int))

    call c_f_pointer(halo_f%nreceives, tmp, [size(halo%receives)])
    do i = 1, size(halo%receives)
       tmp(i) = size(halo%receives(i)%ptr)
    end do

    halo_f%comm = halo%communicator
    halo_f%fluidity_halo = c_loc(halo)
  end function halo_f

  function read_mesh_f(filename, file_format, dim) result(mesh) bind(c)
    character(kind=c_char), intent(in) :: filename(*), file_format(*)
    integer(kind=c_int), value, intent(in) :: dim
    type(mesh_t) :: mesh

    type(vector_field), pointer :: coordinate_field
    type(scalar_field), pointer :: canonical_numbering

    allocate(coordinate_field)

    ! Caution! The canonical numbering descriptor will currently leak. Need
    !  to think about how to do this.
    allocate(canonical_numbering)

    coordinate_field = read_mesh_files(fstring(filename),&
         & fstring(file_format), quad_degree=1, coord_dim=dim)

    if (isparallel()) then
       call read_halos(fstring(filename), coordinate_field)
    end if
    ! Reorder the mesh according to the canonical numbering.
    canonical_numbering=universal_number_field(coordinate_field%mesh)
    ! The canonical numbering's uid is itself.
    canonical_numbering%mesh%uid=>canonical_numbering
    call order_elements(canonical_numbering, coordinate_field)
    coordinate_field%mesh=canonical_numbering%mesh

    call populate_facet_adjacency(coordinate_field%mesh)

    mesh%element_vertex_list = coordinate_field%mesh%ndglno_c
    mesh%dimension = coordinate_field%mesh%shape%cell%dimension
    mesh%cell_vertices = coordinate_field%mesh%shape%cell%entity_counts(0)
    mesh%vertex_count = node_count(coordinate_field)
    mesh%cell_count = element_count(coordinate_field)
    mesh%exterior_facet_count = surface_element_count(coordinate_field)
    mesh%interior_facet_count = &
         (face_count(coordinate_field) - mesh%exterior_facet_count)/2
    mesh%cell_classes = c_loc(coordinate_field%mesh%element_classes)
    mesh%vertex_classes = c_loc(coordinate_field%mesh%node_classes)
    if (associated(coordinate_field%mesh%region_ids)) then
       mesh%region_ids = coordinate_field%mesh%region_ids_c
    else
       mesh%region_ids = C_NULL_PTR
    end if
    mesh%space_dimension = coordinate_field%dim
    mesh%coordinates = coordinate_field%val_c
    mesh%fluidity_coordinate = c_loc(coordinate_field)
    mesh%fluidity_mesh = c_loc(coordinate_field%mesh)

    mesh%exterior_local_facet_number = &
         coordinate_field%mesh%faces%exterior_local_facet_number
    mesh%interior_local_facet_number = &
         coordinate_field%mesh%faces%interior_local_facet_number
    mesh%exterior_facet_cell = &
         coordinate_field%mesh%faces%exterior_facet_cell
    mesh%interior_facet_cell = &
         coordinate_field%mesh%faces%interior_facet_cell

    mesh%boundary_ids = coordinate_field%mesh%faces%boundary_ids_c

  contains

    subroutine populate_facet_adjacency(mesh)
      type(mesh_type), intent(inout) :: mesh

      integer :: facets, efacets, ifacets, f, cell, cell2, facet, c
      integer, pointer :: efacet_num(:), efacet_cell(:)
      integer, pointer :: ifacet_num(:,:), ifacet_cell(:,:)
      integer, pointer :: neigh(:)

      facets = face_count(mesh)
      efacets = surface_element_count(mesh)
      ifacets = (facets - efacets)/2
      assert(2*ifacets == facets - efacets)

      ! Exterior facets.
      mesh%faces%exterior_local_facet_number &
           = malloc(efacets*c_sizeof(1_c_int))
      mesh%faces%exterior_facet_cell = malloc(efacets*c_sizeof(1_c_int))

      call c_f_pointer(mesh%faces%exterior_local_facet_number, efacet_num,&
           & [efacets])
      call c_f_pointer(mesh%faces%exterior_facet_cell, efacet_cell,&
           & [efacets])

      efacet_num = mesh%faces%local_face_number(:efacets) - 1
      efacet_cell = mesh%faces%face_element_list(:efacets) - 1

      ! Interior facets.
      mesh%faces%interior_local_facet_number &
           = malloc(2*ifacets*c_sizeof(1_c_int))
      mesh%faces%interior_facet_cell = malloc(2*ifacets*c_sizeof(1_c_int))

      call c_f_pointer(mesh%faces%interior_local_facet_number, ifacet_num,&
           & [2, ifacets])
      call c_f_pointer(mesh%faces%interior_facet_cell,  ifacet_cell, &
           & [2, ifacets])

      facet = 0

      do cell = 1, element_count(mesh)
         neigh => ele_neigh(mesh, cell)
         do c = 1, size(neigh)
            cell2 = neigh(c)

            ! Do each facet once.
            if (cell2 <= cell) cycle

            facet = facet+1

            f = ele_face(mesh, cell, cell2)
            ifacet_num(1, facet) = local_face_number(mesh, f) - 1
            ifacet_cell(1, facet) = face_ele(mesh, f) - 1

            f = ele_face(mesh, cell2, cell)
            ifacet_num(2, facet) = local_face_number(mesh, f) - 1
            ifacet_cell(2, facet) = face_ele(mesh, f) - 1

         end do

      end do

    end subroutine populate_facet_adjacency

  end function read_mesh_f

  pure function cstringlen(cstring)
    character(kind=c_char), intent(in) :: cstring(*)
    integer :: cstringlen

    cstringlen=0

    do
       if (cstring(cstringlen+1)==C_NULL_CHAR) return
       cstringlen = cstringlen+1
    end do

  end function cstringlen

  pure function fstring(cstring)
    character(kind=c_char), intent(in) :: cstring(*)
    character(cstringlen(cstring)) :: fstring

    integer :: i

    do i = 1, len(fstring)
       fstring(i:i) = cstring(i)
    end do

  end function fstring

  function make_element_shape_from_fiat(fiat_element) result (element)
    type(element_type) :: element
    type(element_t), intent(in) :: fiat_element

    character(len=1024) :: python_string
    integer :: d, e, f, dof
    integer, allocatable :: count(:,:)
    integer, pointer :: dofs_per(:), entity_dofs(:,:)

    element%cell => find_cell(fiat_element%dimension, fiat_element%vertices)

    element%dim = fiat_element%dimension
    element%ndof = fiat_element%ndof
    element%ngi = 0
    element%degree = fiat_element%degree
    element%type = -666 ! A type not of this world.
    call addref(element)

    call c_f_pointer(fiat_element%dofs_per, dofs_per, [element%dim+1])
    call c_f_pointer(fiat_element%entity_dofs, entity_dofs, [3, element%ndof])

    if (dofs_per(element%dim+1) == element%ndof) then
       element%type = ELEMENT_DISCONTINUOUS_LAGRANGIAN
    else
       element%type = ELEMENT_OTHER
    end if

    allocate(element%entity2dofs(0:ubound(element%cell%entities,1),&
         size(element%cell%entities,2)))
    allocate(count(0:ubound(element%cell%entities,1),&
         size(element%cell%entities,2)))

    count = 0

    do d = 0, element%dim
       do e = 1, element%cell%entity_counts(d)
          allocate(element%entity2dofs(d,e)%dofs(dofs_per(d+1)))
       end do
    end do

    do dof=1,element%ndof
       d=entity_dofs(1,dof) ! Dimension of entity
       e=entity_dofs(2,dof)+1 ! Entity
       f=entity_dofs(3,dof)+1 ! DoF number

       count(d,e) = count(d,e)+1
       element%entity2dofs(d,e)%dofs(count(d,e)) = f
    end do

  end function make_element_shape_from_fiat

  subroutine function_space_destructor_f(function_space) bind(c)
    type(c_ptr), value :: function_space

    type(mesh_type), pointer :: mesh

    call c_f_pointer(function_space, mesh)

    call deallocate(mesh)
    deallocate(mesh)

  end subroutine function_space_destructor_f

  subroutine vector_field_destructor_f(field) bind(c)
    type(c_ptr), value :: field

    type(vector_field), pointer :: tmp

    call c_f_pointer(field, tmp)

    call deallocate(tmp)
    deallocate(tmp)

  end subroutine vector_field_destructor_f

end module python_interface_f
