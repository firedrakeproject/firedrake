!    Copyright (C) 2011 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
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
module cell_numbering
  ! A module to enable the numbering of the basic topological entities in
  !  cells.
  !
  ! The following nomenclature will be used:
  !
  ! Entity  Dimension
  !-------------------
  ! Vertex  0
  ! Edge    1
  ! Face    2
  !
  ! Entity Co-dimension
  !---------------------
  ! Facet  1
  ! Cell   0
  !
  ! Topological entities are denoted by an ordered pair of integers with
  ! the first entry being the dimension and the second being the index.
  ! For example, the first vertex would be numbered [0,1] and the third
  ! face would be numbered [2,3]
  !
  ! For a full description, see chapter 5 of the UFC specification.
  use fldebug
  use data_structures
  implicit none

  private

  integer, parameter, public :: CELL_POINT=0, CELL_INTERVAL=1,&
       & CELL_TRIANGLE=2, CELL_QUAD=3, CELL_TET=4,  CELL_HEX=5

  type vertex_list
     integer, dimension(:), allocatable :: vertices
  end type vertex_list

  type cell_type
     ! Identifying parameter of type cell_foo.
     integer :: type
     integer :: dimension
     ! Number of entities of each dimension.
     integer, dimension(0:3) :: entity_counts
     ! Map tuples of vertices to topological entities.
     type(integer_hash_table) :: vertices2entity
     ! Map topological entities to tuples of vertices.
     type(vertex_list), dimension(:,:), allocatable :: entities
     ! Local coordinates of vertices
     real, dimension(:,:), allocatable :: vertex_coords
     ! Corresponding FIAT cell name
     character(len=20) :: fiat_name
  end type cell_type

  type(cell_type), dimension(0:5), target, save :: cells

  public :: cell_type, cells, number_cells, entity_vertices,&
       & vertices_entity, facet_count, find_cell, vertex_count

  logical, save :: initialised=.false.

  interface vertex_count
     module procedure cell_vertex_count
  end interface vertex_count

  interface facet_count
     module procedure cell_facet_count
  end interface facet_count

contains

  pure function cell_vertex_count(cell)
    ! Return the number of verices in a cell
    type(cell_type), intent(in) :: cell
    integer :: cell_vertex_count

    cell_vertex_count = cell%entity_counts(0)

  end function cell_vertex_count

  pure function cell_facet_count(cell)
    ! Return the number of facets in a cell.
    type(cell_type), intent(in) :: cell
    integer :: cell_facet_count

    ! Special case for dim==0
    if (cell%dimension==0) then
       cell_facet_count=0
    else
       cell_facet_count=cell%entity_counts(cell%dimension-1)
    end if

  end function cell_facet_count

  function find_cell(dim, vertices)
    ! Return a pointer to the cell characterised by dim and vertices
    type(cell_type), pointer :: find_cell
    integer, intent(in) :: dim, vertices

    character(len=200) :: errmsg
    integer :: i

    call number_cells

    do i=0,5
       find_cell=>cells(i)
       if (find_cell%dimension==dim &
            .and. find_cell%entity_counts(0)==vertices) then
          return
       end if
    end do

    write(errmsg,&
    "('No cell with dimension ',i0,' and ',i0,' vertices')") dim,&
         & vertices
    FLAbort(errmsg)

  end function find_cell

  function entity_vertices(cell, entity)
    ! Return a pointer to  the list of vertices associated with the given
    ! topological entity.
    type(cell_type), intent(in), target :: cell
    integer, dimension(2), intent(in) :: entity

    integer, dimension(:), pointer :: entity_vertices

    entity_vertices => cell%entities(entity(1), entity(2))%vertices

  end function entity_vertices

  function vertices_entity(cell, vertices)
    ! Return the entity associated with vertices.
    ! Vertices must be in ascending order.
    type(cell_type), intent(in) :: cell
    integer, dimension(:), intent(in) :: vertices

    integer, dimension(2) :: vertices_entity

    vertices_entity=intuncat(fetch(cell%vertices2entity, intcat(vertices)),2)

  end function vertices_entity

  subroutine number_cells
    ! initialisation routine which causes the cells to be populated.

    if (initialised) return
    initialised=.true.

    call number_cell_point
    call number_cell_interval
    call number_cell_triangle
    call number_cell_quad
    call number_cell_tet
    call number_cell_hex

  end subroutine number_cells

  subroutine number_cell_point
    ! Number the topological entities in a triangle
    type(cell_type), pointer :: cell

    cell=> cells(CELL_POINT)

    cell%type=CELL_POINT
    cell%dimension=0
    cell%entity_counts=[1,0,0,0]
    cell%fiat_name = "NoFiatName"

    allocate(cell%entities(0:cell%dimension,maxval(cell%entity_counts)))
    allocate(cell%vertex_coords(cell%entity_counts(0),cell%dimension))
    call allocate(cell%vertices2entity)

    ! Vertex
    call map_vertices_entity(cell, [1], [0,1])

  end subroutine number_cell_point

  subroutine number_cell_interval
    ! Number the topological entities in a triangle
    type(cell_type), pointer :: cell

    cell=> cells(CELL_INTERVAL)

    cell%type=CELL_INTERVAL
    cell%dimension=1
    cell%entity_counts=[2,1,0,0]
    cell%fiat_name = "UFCInterval"

    allocate(cell%entities(0:cell%dimension,maxval(cell%entity_counts)))
    allocate(cell%vertex_coords(cell%entity_counts(0),cell%dimension))
    call allocate(cell%vertices2entity)

    ! Vertices
    ! Note that this numbering treats interval vertices as facets rather
    !  than vertices. This violates UFC. Fix it when the time comes.
    call map_vertices_entity(cell, [1], [0,2])
    call map_vertices_entity(cell, [2], [0,1])
    ! Edge
    call map_vertices_entity(cell, [1,2], [1,1])

    cell%vertex_coords(1,:)=[1]
    cell%vertex_coords(2,:)=[0]

  end subroutine number_cell_interval

  subroutine number_cell_triangle
    ! Number the topological entities in a triangle
    type(cell_type), pointer :: cell

    cell=> cells(CELL_TRIANGLE)

    cell%type=CELL_TRIANGLE
    cell%dimension=2
    cell%entity_counts=[3,3,1,0]
    cell%fiat_name = "UFCTriangle"

    allocate(cell%entities(0:cell%dimension,maxval(cell%entity_counts)))
    allocate(cell%vertex_coords(cell%entity_counts(0),cell%dimension))
    call allocate(cell%vertices2entity)

    ! Vertices
    call map_vertices_entity(cell, [1], [0,1])
    call map_vertices_entity(cell, [2], [0,2])
    call map_vertices_entity(cell, [3], [0,3])
    ! Edges
    call map_vertices_entity(cell, [1,2], [1,3])
    call map_vertices_entity(cell, [1,3], [1,2])
    call map_vertices_entity(cell, [2,3], [1,1])
    ! Face
    call map_vertices_entity(cell, [1,2,3], [2,1])

    cell%vertex_coords(1,:)=[1,0]
    cell%vertex_coords(2,:)=[0,1]
    cell%vertex_coords(3,:)=[0,0]

  end subroutine number_cell_triangle

  subroutine number_cell_quad
    ! Number the topological entities in a quad
    type(cell_type), pointer :: cell

    cell=> cells(CELL_QUAD)

    cell%type=CELL_QUAD
    cell%dimension=2
    cell%entity_counts=[4,4,1,0]
    cell%fiat_name = "NoFiatName"

    allocate(cell%entities(0:cell%dimension,maxval(cell%entity_counts)))
    allocate(cell%vertex_coords(cell%entity_counts(0),cell%dimension))
    call allocate(cell%vertices2entity)

    ! Vertices
    call map_vertices_entity(cell, [1], [0,1])
    call map_vertices_entity(cell, [2], [0,2])
    call map_vertices_entity(cell, [3], [0,3])
    call map_vertices_entity(cell, [4], [0,4])
    ! Edges
    call map_vertices_entity(cell, [3,4], [1,1])
    call map_vertices_entity(cell, [2,4], [1,2])
    call map_vertices_entity(cell, [1,3], [1,3])
    call map_vertices_entity(cell, [1,2], [1,4])
    ! Face
    call map_vertices_entity(cell, [1,2,3,4], [2,1])

    cell%vertex_coords(1,:)=[0,0]
    cell%vertex_coords(2,:)=[1,0]
    cell%vertex_coords(3,:)=[0,1]
    cell%vertex_coords(4,:)=[1,1]

  end subroutine number_cell_quad

  subroutine number_cell_tet
    ! Number the topological entities in a tet
    type(cell_type), pointer :: cell

    cell=> cells(CELL_TET)

    cell%type=CELL_TET
    cell%dimension=3
    cell%entity_counts=[4,6,4,1]
    cell%fiat_name = "UFCTetrahedron"

    allocate(cell%entities(0:cell%dimension,maxval(cell%entity_counts)))
    allocate(cell%vertex_coords(cell%entity_counts(0),cell%dimension))
    call allocate(cell%vertices2entity)

    ! Vertices
    call map_vertices_entity(cell, [1], [0,1])
    call map_vertices_entity(cell, [2], [0,2])
    call map_vertices_entity(cell, [3], [0,3])
    call map_vertices_entity(cell, [4], [0,4])
    ! Edges
    call map_vertices_entity(cell, [3,4], [1,1])
    call map_vertices_entity(cell, [2,4], [1,2])
    call map_vertices_entity(cell, [2,3], [1,3])
    call map_vertices_entity(cell, [1,4], [1,4])
    call map_vertices_entity(cell, [1,3], [1,5])
    call map_vertices_entity(cell, [1,2], [1,6])
    ! Faces
    call map_vertices_entity(cell, [2,3,4], [2,1])
    call map_vertices_entity(cell, [1,3,4], [2,2])
    call map_vertices_entity(cell, [1,2,4], [2,3])
    call map_vertices_entity(cell, [1,2,3], [2,4])
    ! Cell
    call map_vertices_entity(cell, [1,2,3,4], [3,1])

    cell%vertex_coords(1,:)=[1,0,0]
    cell%vertex_coords(2,:)=[0,1,0]
    cell%vertex_coords(3,:)=[0,0,1]
    cell%vertex_coords(4,:)=[0,0,0]

  end subroutine number_cell_tet

  subroutine number_cell_hex
    ! Number the topological entities in a hex
    type(cell_type), pointer :: cell

    cell=> cells(CELL_HEX)

    cell%type=CELL_HEX
    cell%dimension=3
    cell%entity_counts=[8,12,6,1]
    cell%fiat_name = "UFCNoFiatName"

    allocate(cell%entities(0:cell%dimension,maxval(cell%entity_counts)))
    allocate(cell%vertex_coords(cell%entity_counts(0),cell%dimension))
    call allocate(cell%vertices2entity)

    ! Vertices
    call map_vertices_entity(cell, [1], [0,1])
    call map_vertices_entity(cell, [2], [0,2])
    call map_vertices_entity(cell, [3], [0,3])
    call map_vertices_entity(cell, [4], [0,4])
    call map_vertices_entity(cell, [5], [0,5])
    call map_vertices_entity(cell, [6], [0,6])
    call map_vertices_entity(cell, [7], [0,7])
    call map_vertices_entity(cell, [8], [0,8])
    ! Edges
    call map_vertices_entity(cell, [7,8], [1,1])
    call map_vertices_entity(cell, [6,8], [1,2])
    call map_vertices_entity(cell, [5,7], [1,3])
    call map_vertices_entity(cell, [5,6], [1,4])
    call map_vertices_entity(cell, [4,8], [1,5])
    call map_vertices_entity(cell, [3,7], [1,6])
    call map_vertices_entity(cell, [3,4], [1,7])
    call map_vertices_entity(cell, [2,6], [1,8])
    call map_vertices_entity(cell, [2,4], [1,9])
    call map_vertices_entity(cell, [1,5], [1,10])
    call map_vertices_entity(cell, [1,3], [1,11])
    call map_vertices_entity(cell, [1,2], [1,12])
    ! Faces
    call map_vertices_entity(cell, [5,6,7,8], [2,1])
    call map_vertices_entity(cell, [3,4,7,8], [2,2])
    call map_vertices_entity(cell, [2,4,6,8], [2,3])
    call map_vertices_entity(cell, [1,3,5,7], [2,4])
    call map_vertices_entity(cell, [1,2,5,6], [2,5])
    call map_vertices_entity(cell, [1,2,3,4], [2,6])
    ! Cell
    call map_vertices_entity(cell, [1,2,3,4,5,6,7,8], [3,1])

    cell%vertex_coords(1,:)=[0,0,0]
    cell%vertex_coords(2,:)=[1,0,0]
    cell%vertex_coords(3,:)=[0,1,0]
    cell%vertex_coords(4,:)=[1,1,0]
    cell%vertex_coords(5,:)=[0,0,1]
    cell%vertex_coords(6,:)=[1,0,1]
    cell%vertex_coords(7,:)=[0,1,1]
    cell%vertex_coords(8,:)=[1,1,1]

  end subroutine number_cell_hex

  subroutine map_vertices_entity(cell, vertices, entity)
    ! Set the entity-vertices and vertices-entity maps of cell.
    type(cell_type), intent(inout) :: cell
    integer, dimension(:), intent(in) :: vertices
    integer, dimension(2), intent(in) :: entity

    call insert(cell%vertices2entity, intcat(vertices), intcat(entity))

    ! Fortran 2003 says this line is not necessary.
    allocate(cell%entities(entity(1), entity(2))%vertices(size(vertices)))

    cell%entities(entity(1), entity(2))%vertices=vertices

  end subroutine map_vertices_entity

  function intcat(tuple)
    ! Concatenate a tuple of integers using base 16 digit
    ! concatentation. Base 16 is used because it is the first power of 2
    ! greater than 12 which is the largest number we need to handle.
    integer, dimension(:), intent(in) :: tuple
    integer :: intcat

    integer :: i

    assert(all(tuple<16))

    intcat=0
    do i = 1,size(tuple)
       intcat=16*intcat+tuple(i)
    end do

  end function intcat

  function intuncat(int, n)
    ! Split a concatenated integer into n parts.
    integer, intent(in) :: int, n
    integer, dimension(n) :: intuncat

    integer :: tmpint, i

    intuncat=0
    tmpint=int

    do i=n,1,-1
       intuncat(i)=mod(tmpint, 16)
       tmpint=tmpint/16
    end do

  end function intuncat

end module cell_numbering
