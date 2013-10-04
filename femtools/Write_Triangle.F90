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

module write_triangle

  use elements
  use fields
  use state_module
  use futils
  use parallel_tools
  use field_options

  implicit none

  private

  public :: write_triangle_files

  interface write_triangle_files
    module procedure write_mesh_to_triangles, &
      & write_positions_to_triangles
  end interface write_triangle_files

contains

  subroutine write_mesh_to_triangles(filename, state, mesh)
    !!< Write out the supplied mesh to the specified filename as triangle files.

    character(len = *), intent(in) :: filename
    type(state_type), intent(in) :: state
    type(mesh_type), intent(in) :: mesh

    type(vector_field):: positions

    ! gets a coordinate field for the given mesh, if necessary
    ! interpolated from "Coordinate", always takes a reference:
    positions = get_nodal_coordinate_field(state, mesh)

    call write_triangle_files(filename, positions)

    call deallocate(positions)

  end subroutine write_mesh_to_triangles

  subroutine write_positions_to_triangles(filename, positions, print_internal_faces)
    !!< Write out the mesh given by the position field in triangle files:
    !!<    a .node and a .ele-file (and a .face file if the mesh has a %faces
    !!<    component with more than 0 surface elements)
    !!< In parallel, empty trailing processes are not written.
    character(len=*), intent(in):: filename
    type(vector_field), intent(in):: positions
    logical, intent(in), optional :: print_internal_faces

    integer :: nparts

    ! How many processes contain data?
    nparts = get_active_nparts(ele_count(positions))

    ! Write out data only for those processes that contain data - SPMD requires
    ! that there be no early return

    if(getprocno() <= nparts) then

       ! write .node file with columns if present
       if (associated(positions%mesh%columns)) then
         call write_triangle_node_file_with_columns(filename, positions)
       else
         call write_triangle_node_file(filename, positions)
       end if

       call write_triangle_ele_file(filename, positions%mesh)
    end if

    if (present_and_true(print_internal_faces) .and. .not. has_faces(positions%mesh)) then
       call add_faces(positions%mesh)
    end if

    if(getprocno() <= nparts) then
       if (present_and_true(print_internal_faces)) then
          call write_triangle_face_file_full(filename, positions%mesh)
       else
          call write_triangle_face_file(filename, positions%mesh)
       end if
    end if

  end subroutine write_positions_to_triangles

  subroutine write_triangle_node_file(filename, field)
    !!< Writes out .node-file for the given position field
    character(len=*), intent(in):: filename
    type(vector_field), intent(in):: field

    character(len = 7 + int2str_len(huge(0)) + real_format_len(padding = 1)) :: format_buffer
    integer unit, nodes, dim, no_coords, i

    unit=free_unit()

    nodes=node_count(field)
    dim=mesh_dim(field)
    no_coords=field%dim

    open(unit=unit, file=trim(filename)//'.node', action='write', err=41)

    ! header line: nodes, dim, no attributes, no boundary markers
    write(unit, *, err=42) nodes, dim, 0, 0

    format_buffer = "(i0,a," // int2str(no_coords) // real_format(padding = 1) // ")"
    do i=1, nodes
       write(unit, trim(format_buffer), err=42) i, " ", node_val(field, i)
    end do

    close(unit=unit, err=43)
    ! succesful return
    return

41  FLExit("Failed to open .node file for writing.")

42  FLExit("Error while writing .node file.")

43  FLExit("Failed to close .node file for writing.")

  end subroutine write_triangle_node_file

  subroutine write_triangle_node_file_with_columns(filename, field)
    !!< Writes out .node-file for the given position field
    !!< Write field%mesh%columns as node attribute
    character(len=*), intent(in):: filename
    type(vector_field), intent(in):: field

    character(len = 12 + int2str_len(huge(0)) + real_format_len(padding = 1)) :: format_buffer
    integer unit, nodes, dim, i

    unit=free_unit()

    nodes=node_count(field)
    dim=mesh_dim(field)

    open(unit=unit, file=trim(filename)//'.node', action='write', err=41)

    ! header line: nodes, dim, no attributes (=1 for columns), no boundary markers
    write(unit, *, err=42) nodes, dim, 1, 0

    format_buffer = "(i0,a," // int2str(dim) // real_format(padding = 1) // ",a,i0)"
    do i=1, nodes
       write(unit, trim(format_buffer), err=42) i, " ", node_val(field, i), " ", field%mesh%columns(i)
    end do

    close(unit=unit, err=43)
    ! succesful return
    return

41  FLExit("Failed to open .node file for writing.")

42  FLExit("Error while writing .node file.")

43  FLExit("Failed to close .node file for writing.")

  end subroutine write_triangle_node_file_with_columns

  subroutine write_triangle_ele_file(filename, mesh)
    !!< Writes out .ele-file for the given mesh
    character(len=*), intent(in):: filename
    type(mesh_type), intent(in):: mesh

    integer unit, elements, nloc, i

    unit=free_unit()

    elements=ele_count(mesh)
    nloc=ele_loc(mesh, 1)

    open(unit=unit, file=trim(filename)//'.ele', action='write', err=41)

    ! Currently region_ids are lost in adapts (can remove the associated tests
    ! when this is fixed)
    if(associated(mesh%region_ids)) then
       ! header line: elements, nloc, 1 attribute
       write(unit, *, err=42) elements, nloc, 1
    else
       ! header line: elements, nloc, no attributes
       write(unit, *, err=42) elements, nloc, 0
    end if

    do i=1, elements
       if(associated(mesh%region_ids)) then
          write(unit, *, err=42) i, ele_nodes(mesh, i), ele_region_id(mesh, i)
       else
          write(unit, *, err=42) i, ele_nodes(mesh, i)
       end if
    end do

    close(unit=unit, err=43)
    ! succesful return
    return

41  FLExit("Failed to open .ele file for writing.")

42  FLExit("Error while writing .ele file.")

43  FLExit("Failed to close .ele file for writing.")

  end subroutine write_triangle_ele_file

  subroutine write_triangle_face_file_full(filename, mesh)
    !!< Writes out .face-file for the given mesh
    character(len=*), intent(in):: filename
    type(mesh_type), intent(in):: mesh

    integer unit, dim, nofaces, i
    integer :: stotel, dg_total ! dg_total counts each internal face twice
    integer :: ele, neigh, j, face
    integer, dimension(:), pointer :: neighbours

    unit=free_unit()

    dim=mesh_dim(mesh)
    stotel=surface_element_count(mesh)
    dg_total=size(mesh%faces%face_list%sparsity%colm)

    nofaces = (dg_total - stotel) / 2 ! internal faces, only once
    nofaces = nofaces + stotel ! and the surface mesh

    select case(dim)
      case(3)
        open(unit=unit, file=trim(filename)//'.face', action='write', err=41)
      case(2)
        open(unit=unit, file=trim(filename)//'.edge', action='write', err=41)
      case(1)
        open(unit=unit, file=trim(filename)//'.bound', action='write', err=41)
      case default
        ewrite(-1, "(a,i0)") "For dimension ", dim
        FLAbort("Invalid dimension")
    end select

    ! header line: nofaces, no boundary marker
    write(unit, *, err=42) nofaces, 0

    i = 1
    do ele=1,ele_count(mesh)
       neighbours => ele_neigh(mesh, ele)
       do j=1,size(neighbours)
          neigh = neighbours(j)
          ! we need to not count the internal face twice
          ! so only print out the internal face for the ele < neigh
          ! case; thus we don't print out the neigh > ele case
          ! and we only do it once.
          if (neigh < ele) then
             face = ele_face(mesh, ele, neigh)
             write(unit, *, err=42) i, face_global_nodes(mesh, face), 0
             i = i + 1
          end if
       end do
    end do

    close(unit=unit, err=43)
    ! succesful return
    return

41  FLExit("Failed to open .face/.edge/.bound file for writing.")

42  FLExit("Error while writing .face/.edge/.bound file.")

43  FLExit("Failed to close .face/.edge/.bound file for writing.")

  end subroutine write_triangle_face_file_full

  subroutine write_triangle_face_file(filename, mesh)
    !!< Writes out .face-file for the given mesh
    character(len=*), intent(in):: filename
    type(mesh_type), intent(in):: mesh

    integer :: unit, dim, nofaces, i
    integer :: nolabels

    unit=free_unit()

    dim=mesh_dim(mesh)
    nofaces=surface_element_count(mesh)

    select case(dim)
      case(3)
        open(unit=unit, file=trim(filename)//'.face', action='write', err=41)
      case(2)
        open(unit=unit, file=trim(filename)//'.edge', action='write', err=41)
      case(1)
        open(unit=unit, file=trim(filename)//'.bound', action='write', err=41)
      case default
        ewrite(-1, "(a,i0)") "For dimension ", dim
        FLAbort("Invalid dimension")
    end select

    if (has_internal_boundaries(mesh)) then
      ! If the mesh is periodic, we want to write out the parent element of every face
      nolabels = 2
    else
      nolabels = 1
    end if

    ! header line: nofaces, and number of boundary markers
    write(unit, *, err=42) nofaces, nolabels

    if (.not. has_internal_boundaries(mesh)) then
      do i=1, nofaces
         write(unit, *, err=42) i, face_global_nodes(mesh, i), &
              surface_element_id(mesh, i)
      end do
    else
      do i=1, nofaces
         write(unit, *, err=42) i, face_global_nodes(mesh, i), &
              surface_element_id(mesh, i), face_ele(mesh, i)
      end do
    end if

    close(unit=unit, err=43)
    ! succesful return
    return

41  FLExit("Failed to open .face/.edge/.bound file for writing.")

42  FLExit("Error while writing .face/.edge/.bound file.")

43  FLExit("Failed to close .face/.edge/.bound file for writing.")

  end subroutine write_triangle_face_file

end module write_triangle
