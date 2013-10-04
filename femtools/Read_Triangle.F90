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

module read_triangle
  !!< This module reads triangle files and results in a vector field of
  !!< positions.

  use futils
  use elements
  use fields
  use state_module
  use spud

  implicit none

  private

  interface read_triangle_files
     module procedure read_triangle_files_to_field, read_triangle_simple
  end interface

  public :: read_triangle_files, identify_triangle_file, read_elemental_mappings, read_triangle_serial

contains

  subroutine identify_triangle_file(filename, dim, loc, nodes, elements, &
       node_attributes, selements, selement_boundaries)
    !!< Discover the dimension and size of the triangle inputs.
    !!< Filename is the base name of the triangle file without .node or .ele.
    !!< In parallel, filename must *include* the process number.

    character(len=*), intent(in) :: filename
    !! Dimension of mesh elements.
    integer, intent(out), optional :: dim
    !! Number of vertices of elements.
    integer, intent(out), optional :: loc
    !! Node and element counts.
    integer, intent(out), optional :: nodes, elements
    integer, intent(out), optional :: node_attributes
    ! Surface element meta data
    integer, optional, intent(out) :: selements
    integer, optional, intent(out) :: selement_boundaries

    integer :: node_unit, element_unit, selement_unit
    integer :: lnodes, ldim, lnode_attributes, node_boundaries
    integer :: lelements, lloc, ele_attributes
    integer :: lselements, lselement_boundaries
    logical :: file_exists

    ! Read node file header
    inquire(file = trim(filename) // ".node", exist = file_exists)
    if(.not. file_exists) then
      ewrite(-1, *) "For triangle file with base name " // trim(filename)
      FLExit(".node file not found")
    end if
    ewrite(2, *) "Opening " // trim(filename) // ".node for reading."
    node_unit = free_unit()
    open(unit = node_unit, file = trim(filename) // ".node", err = 42, action = "read")
    read (node_unit, *) lnodes, ldim, lnode_attributes, node_boundaries
    close(node_unit)

    ! Read volume element file header
    lelements = 0
    lloc = 0
    ele_attributes = 0
    inquire(file = trim(filename) // ".ele", exist = file_exists)
    if(file_exists) then
      ewrite(2, *) "Opening " // trim(filename) // ".ele for reading"
      element_unit = free_unit()
      open(unit = element_unit, file = trim(filename) // ".ele", err = 43, action = "read")
      read (element_unit, *) lelements, lloc, ele_attributes
      close(element_unit)
    else if (present(loc) .or. present(elements)) then
      ewrite(-1, *) "For triangle file with base name " // trim(filename)
      FLExit(".ele file not found")
    end if

    ! Read the surface element file header
    lselements = 0
    lselement_boundaries = 0
    select case(ldim)
      case(1)
        inquire(file = trim(filename) // ".bound", exist = file_exists)
        if(file_exists) then
          ewrite(2, *) "Opening " // trim(filename) // ".bound for reading"
          selement_unit = free_unit()
          open(unit = selement_unit, file = trim(filename) // ".bound", err = 44, action = "read")
          read(selement_unit, *) lselements, lselement_boundaries
          close(selement_unit)
        end if
      case(2)
        inquire(file = trim(filename) // ".edge", exist = file_exists)
        if(file_exists) then
          ewrite(2, *) "Opening " // trim(filename) // ".edge for reading"
          selement_unit = free_unit()
          open(unit = selement_unit, file = trim(filename) // ".edge", err = 45, action = "read")
          read(selement_unit, *) lselements, lselement_boundaries
          close(selement_unit)
        end if
      case(3)
        inquire(file = trim(filename) // ".face", exist = file_exists)
        if(file_exists) then
          ewrite(2, *) "Opening " // trim(filename) // ".face for reading"
          selement_unit = free_unit()
          open(unit = selement_unit, file = trim(filename) // ".face", err = 46, action = "read")
          read(selement_unit, *) lselements, lselement_boundaries
          close(selement_unit)
        end if
    end select

    if(present(nodes)) then
       nodes = lnodes
    end if
    if(present(dim)) then
       dim = ldim
    end if
    if(present(node_attributes)) then
       node_attributes = lnode_attributes
    end if
    if(present(loc)) then
       loc = lloc
    end if
    if(present(elements)) then
       elements = lelements
    end if
    if(present(selements)) then
      selements = lselements
    end if
    if(present(selement_boundaries)) then
      selement_boundaries = lselement_boundaries
    end if

    return

42  FLExit("Unable to open "//trim(filename)//".node")

43  FLExit("Unable to open "//trim(filename)//".ele")

44  FLExit("Unable to open " // trim(filename) // ".bound")

45  FLExit("Unable to open " // trim(filename) // ".edge")

46  FLExit("Unable to open " // trim(filename) // ".face")

  end subroutine identify_triangle_file

  function read_triangle_files_to_field(filename, shape) result (field)
    !!< Filename is the base name of the triangle file without .node or .ele .
    !!< In parallel the filename must *not* include the process number.

    character(len=*), intent(in) :: filename
    type(element_type), intent(in), target :: shape
    type(vector_field)  :: field

    integer :: node_unit, ele_unit
    real, allocatable, dimension(:) :: read_buffer
    integer, allocatable, dimension(:,:) :: edge_buffer
    integer, allocatable, dimension(:) :: sndglno
    integer, allocatable, dimension(:) :: boundary_ids, element_owner

    character(len = parallel_filename_len(filename)) :: lfilename
    integer :: i, j, nodes, dim, xdim, node_attributes, boundaries,&
         & ele_attributes, loc, sloc, elements, edges, edge_count
    integer, allocatable, dimension(:):: node_order
    logical :: file_exists
    type(mesh_type) :: mesh

    ! If running in parallel, add the process number
    if(isparallel()) then
      lfilename = parallel_filename(filename)
    else
      lfilename = trim(filename)
    end if

    node_unit=free_unit()

    ewrite(2, *) "Opening "//trim(lfilename)//".node for reading."
    ! Open node file
    open(unit=node_unit, file=trim(lfilename)//".node", err=42, action="read")

    ! Read node file header.
    read (node_unit, *) nodes, xdim, node_attributes, boundaries

    ele_unit=free_unit()

    ewrite(2, *) "Opening "//trim(lfilename)//".ele for reading."
    ! Open element file
    open(unit=ele_unit, file=trim(lfilename)//".ele", err=43, action="read")

    ! Read element file header.
    read (ele_unit, *) elements, loc, ele_attributes

    assert(loc==shape%ndof)
    allocate(node_order(loc))
    select case(loc)
    case(3)
       node_order = (/1,2,3/)
    case(6)
       node_order = (/1,6,2,5,4,3/)
    case default
       do j=1,loc
          node_order(j)=j
       end do
    end select

    call allocate(mesh, nodes, elements, shape, name="CoordinateMesh")

    if ((xdim==2).and.(have_option('/geometry/spherical_earth/'))) then
      call allocate(field, xdim+1, mesh, name="Coordinate") ! Pseudo 2D mesh points have 3 coordinates
    else
       call allocate(field, xdim, mesh, name="Coordinate")
    end if

    ! Drop the local reference to mesh - now field owns the only reference.
    call deallocate(mesh)

    if ((xdim==2).and.(have_option('/geometry/spherical_earth/'))) then
      allocate(read_buffer(xdim+node_attributes+boundaries+2))
    else
      allocate(read_buffer(xdim+node_attributes+boundaries+1))
    end if

    if(node_attributes==1) then ! this assumes the node attribute are column numbers
      allocate(field%mesh%columns(1:nodes))
   end if

    do i=1,nodes
       if ((xdim==2).and.(have_option('/geometry/spherical_earth/'))) then
         read(node_unit,*) read_buffer
         forall (j=1:xdim+1)
            field%val(j,i)=read_buffer(j+1)
         end forall
         if (node_attributes==1) then
           field%mesh%columns(i)=floor(read_buffer(xdim+2))
         end if
       else
         read(node_unit,*) read_buffer
         forall (j=1:xdim)
            field%val(j,i)=read_buffer(j+1)
         end forall
         if (node_attributes==1) then
           field%mesh%columns(i)=floor(read_buffer(xdim+2))
         end if
       end if
    end do

    deallocate(read_buffer)
    allocate(read_buffer(loc+ele_attributes+1))

    if(ele_attributes==1) then  ! this assumes that the element attribute is a region id
      call allocate_region_ids(field%mesh, elements)
    end if

    do i=1,elements
       read(ele_unit,*) read_buffer
       field%mesh%ndglno((i-1)*loc+1:i*loc)=floor(read_buffer(node_order+1))
       if(ele_attributes==1) then
          field%mesh%region_ids(i)=read_buffer(loc+2)
       end if
    end do

    close(node_unit)
    close(ele_unit)

    ! Get the mesh dimension so we know which files to look for
    dim=shape%dim

    ! Open edge file
    select case (dim)
    case(1)
       inquire(file=trim(lfilename)//".bound",exist=file_exists)
       if(file_exists) then
          ewrite(2, *) "Opening "//trim(lfilename)//".bound for reading."
          open(unit=node_unit, file=trim(lfilename)//".bound", err=41, &
                action="read")
       end if
    case(2)
       inquire(file=trim(lfilename)//".edge",exist=file_exists)
       if(file_exists) then
          ewrite(2, *) "Opening "//trim(lfilename)//".edge for reading."
          open(unit=node_unit, file=trim(lfilename)//".edge", err=41, &
                action="read")
       end if
    case(3)
       inquire(file=trim(lfilename)//".face",exist=file_exists)
       if(file_exists) then
          ewrite(2, *) "Opening "//trim(lfilename)//".face for reading."
          open(unit=node_unit, file=trim(lfilename)//".face", err=41, &
                action="read")
       end if
    end select

    if(file_exists) then
      ! Read edge file header.
      read (node_unit, *) edges, boundaries
    else
      edges = 0
      boundaries = 1
    end if

    if(edges==0) then
       file_exists = .false.
       close(node_unit)
    end if

    select case(cell_family(shape))
    case(FAMILY_SIMPLEX)
      if ((loc/=dim+1).and.(boundaries/=0)) then
        ewrite(0,*) "Warning: triangle boundary markers not supported for qua", &
              &"dratic space elements."
        if(file_exists) then
            file_exists= .false.
            close(node_unit)
        end if
      end if
      sloc=loc-1
    case(FAMILY_CUBE)
      sloc=loc/2
    case default
       ewrite(-1,*) "While reading triangle files with basename "//trim(lfilename)
       FLAbort('Illegal element family')
    end select

    allocate(edge_buffer(sloc+boundaries+1,edges))
    edge_buffer=0
    allocate(sndglno(edges*sloc))
    sndglno=0
    allocate(boundary_ids(1:edges))
    boundary_ids=0
    if (boundaries==2) then
      allocate(element_owner(1:edges))
      element_owner=0
    end if
    edge_count=0

    if (boundaries==0) then
       ewrite(0,*) "Warning: triangle edge file has no boundary markers"
       if(file_exists) then
          file_exists=.false.
          close(node_unit)
       end if
    else
      if(file_exists) then
        read(node_unit, *) edge_buffer

        do i=1, edges
          if (edge_buffer(sloc+2,i)/=0) then
            ! boundary edge/face
            edge_count=edge_count+1
            sndglno((edge_count-1)*sloc+1:edge_count*sloc)= &
              edge_buffer(2:sloc+1,i)
            boundary_ids(edge_count)=edge_buffer(sloc+2,i)
            if (boundaries==2) then
              element_owner(edge_count)=edge_buffer(sloc+3,i)
            end if
          end if
        end do

        file_exists=.false.
        close(node_unit)
      end if
    end if

    if (boundaries<2) then
      call add_faces(field%mesh, &
          &               sndgln=sndglno(1:edge_count*sloc), &
          &               boundary_ids=boundary_ids(1:edge_count))
    else
      call add_faces(field%mesh, &
          &               sndgln=sndglno(1:edge_count*sloc), &
          &               boundary_ids=boundary_ids(1:edge_count), &
          &               element_owner=element_owner)
    end if

    deallocate(edge_buffer)
    deallocate(sndglno)
    deallocate(boundary_ids)

41  continue ! We jump to here if there was no edge file.

    return

42  FLExit("Unable to open "//trim(lfilename)//".node")

43  FLExit("Unable to open "//trim(lfilename)//".ele")


  end function read_triangle_files_to_field

  function read_triangle_simple(filename, quad_degree, quad_ngi, no_faces, quad_family, mdim) result (field)
    !!< A simpler mechanism for reading a triangle file into a field.
    !!< In parallel the filename must *not* include the process number.

    character(len=*), intent(in) :: filename
    !! The degree of the quadrature.
    integer, intent(in), optional, target :: quad_degree
    !! The degree of the quadrature.
    integer, intent(in), optional, target :: quad_ngi
    !! Whether to add_faces on the resulting mesh.
    logical, intent(in), optional :: no_faces
    !! What quadrature family to use
    integer, intent(in), optional :: quad_family
    !! Dimension of mesh
    integer, intent(in), optional :: mdim

    type(vector_field) :: field
    type(quadrature_type) :: quad
    type(element_type) :: shape

    integer :: dim, loc

    if(isparallel()) then
      call identify_triangle_file(parallel_filename(filename), dim, loc)
    else
      call identify_triangle_file(filename, dim, loc)
    end if

    if (present(mdim)) then
       dim=mdim
    end if

    if (present(quad_degree)) then
      quad=make_quadrature(loc, dim, degree=quad_degree, family=quad_family)
    else if (present(quad_ngi)) then
      quad=make_quadrature(loc, dim, ngi=quad_ngi, family=quad_family)
    else
      FLAbort("Need to specify either quadrature degree or ngi")
    end if

    shape=make_element_shape(loc, dim, 1, quad)

    if (present_and_true(no_faces)) then
      field=read_triangle_files_to_field_no_faces(filename, shape)
    else
      field=read_triangle_files(filename, shape)
    end if

    ! deallocate our references of shape and quadrature:
    ! NOTE: we're using the specific deallocate interface here
    !       to make the intel compiler shut up
    call deallocate_element(shape)
    call deallocate(quad)

  end function read_triangle_simple

  function read_elemental_mappings(positions, filename, map, stat) result(field)
    type(vector_field), intent(in) :: positions
    character(len=*), intent(in) :: filename, map
    type(scalar_field) :: field
    integer, optional, intent(out) :: stat

    integer :: elements, unit, ele, current_element, target_element
    integer :: io
    real :: t

    if(present(stat)) stat = 0

    if(isparallel()) then
      call identify_triangle_file(parallel_filename(filename), elements=elements)
    else
      call identify_triangle_file(filename, elements=elements)
    end if

    unit = free_unit()
    open(unit=unit, file=trim(filename)// "." // trim(map), action="read", iostat=io, status="old")
    if (io == 0) then
      field = piecewise_constant_field(positions%mesh, trim(map))

      do ele=1,elements
        read (unit, *) current_element, target_element
        t = target_element
        call set(field, current_element, t)
      end do
    else
      if(present(stat)) then
        stat = io
        return
      else
         ewrite(-1,*) "While opening "//trim(filename)// "." // trim(map)
        FLAbort("Failed to read elemental mappings")
      end if
    end if

    close(unit)

  end function read_elemental_mappings

  ! I'm sorry -- I had to copy this function.
  ! The add_faces call on the triangle files I am using
  ! crashes, because I am using the edge markers for
  ! something different to defining boundary labels.
  ! However, I couldn't add an optional logical,
  ! as then that makes the interface indistinguishable
  ! from read_triangle_files_to_state!
  ! -- pfarrell

  function read_triangle_files_to_field_no_faces(filename, shape) result (field)
    !!< Filename is the base name of the triangle file without .node or .ele .
    !!< In parallel the filename must *not* include the process number.

    character(len=*), intent(in) :: filename
    type(element_type), intent(in), target :: shape
    type(vector_field)  :: field

    integer :: node_unit, ele_unit
    real, allocatable, dimension(:) :: read_buffer
    integer, allocatable, dimension(:,:) :: edge_buffer
    integer, allocatable, dimension(:) :: sndglno
    integer, allocatable, dimension(:) :: boundary_ids

    character(len = parallel_filename_len(filename)) :: lfilename
    integer :: i, j, nodes, dim, xdim, node_attributes, boundaries,&
         & ele_attributes, loc, sloc, elements, edges, edge_count
    integer, allocatable, dimension(:):: node_order
    logical :: file_exists
    type(mesh_type) :: mesh

    ! If running in parallel, add the process number
    if(isparallel()) then
      lfilename = parallel_filename(filename)
    else
      lfilename = trim(filename)
    end if

    node_unit=free_unit()

    ewrite(2, *) "Opening "//trim(lfilename)//".node for reading."
    ! Open node file
    open(unit=node_unit, file=trim(lfilename)//".node", err=42, action="read")

    ! Read node file header.
    read (node_unit, *) nodes, xdim, node_attributes, boundaries

    ele_unit=free_unit()

    ewrite(2, *) "Opening "//trim(lfilename)//".ele for reading."
    ! Open element file
    open(unit=ele_unit, file=trim(lfilename)//".ele", err=43, action="read")

    ! Read element file header.
    read (ele_unit, *) elements, loc, ele_attributes

    assert(loc==shape%ndof)
    allocate(node_order(loc))
    select case(loc)
    case(3)
       node_order = (/1,2,3/)
    case(6)
       node_order = (/1,6,2,5,4,3/)
    case default
       do j=1,loc
          node_order(j)=j
       end do
    end select

    call allocate(mesh, nodes, elements, shape, name=filename)

    ! Field has an upper index of 3. Therefore, if dim==3 and
    ! node_attributes>0 then we get an out of bounds reference. Assume
    ! here that when there are node attributes they can be ignored.
    call allocate(field, xdim, mesh, name="Coordinate")

    ! Drop the local reference to mesh - now field owns the only reference.
    call deallocate(mesh)

    allocate(read_buffer(xdim+node_attributes+boundaries+1))

    do i=1,nodes
       read(node_unit,*) read_buffer
       forall (j=1:xdim)
          field%val(j,i)=read_buffer(j+1)
       end forall
    end do

    deallocate(read_buffer)
    allocate(read_buffer(loc+ele_attributes+1))

    if(ele_attributes==1) then  ! this assumes that the element attribute is a region id
      call allocate_region_ids(field%mesh, elements)
    end if

    do i=1,elements
       read(ele_unit,*) read_buffer
       field%mesh%ndglno((i-1)*loc+1:i*loc)=floor(read_buffer(node_order+1))
       if(ele_attributes==1) then
          field%mesh%region_ids(i)=read_buffer(loc+2)
       end if
    end do

    close(node_unit)
    close(ele_unit)

    ! Get the mesh dimension so we know which files to look for
    dim=shape%dim

    ! Open edge file
    select case (dim)
    case(1)
       inquire(file=trim(lfilename)//".bound",exist=file_exists)
       if(file_exists) then
          ewrite(2, *) "Opening "//trim(lfilename)//".bound for reading."
          open(unit=node_unit, file=trim(lfilename)//".bound", err=41, &
                action="read")
       end if
    case(2)
       inquire(file=trim(lfilename)//".edge",exist=file_exists)
       if(file_exists) then
          ewrite(2, *) "Opening "//trim(lfilename)//".edge for reading."
          open(unit=node_unit, file=trim(lfilename)//".edge", err=41, &
                action="read")
       end if
    case(3)
       inquire(file=trim(lfilename)//".face",exist=file_exists)
       if(file_exists) then
          ewrite(2, *) "Opening "//trim(lfilename)//".face for reading."
          open(unit=node_unit, file=trim(lfilename)//".face", err=41, &
                action="read")
       end if
    end select

    if(file_exists) then
      ! Read edge file header.
      read (node_unit, *) edges, boundaries
    else
      edges = 0
      boundaries = 1
    end if

    if (boundaries==0 .or. edges==0) then
       if(file_exists) then
          close(node_unit)
       end if
       goto 41
    end if

    select case(cell_family(shape))
    case(FAMILY_SIMPLEX)
      if (loc/=dim+1) then
         ewrite(0,*) "Warning: triangle boundary markers not supported for qua",&
               &"dratic space elements."
         if(file_exists) then
            close(node_unit)
         end if
         goto 41
      end if
      sloc=loc-1
    case(FAMILY_CUBE)
      sloc=loc/2
    case default
      FLAbort('Illegal element family')
    end select
    allocate(edge_buffer(sloc+boundaries+1,edges))
    allocate(sndglno(edges*sloc))
    allocate(boundary_ids(1:edges))

    if(file_exists) then
      read(node_unit, *) edge_buffer
    end if

    edge_count=0
    do i=1, edges
       if (edge_buffer(sloc+2,i)/=0) then
         ! boundary edge/face
         edge_count=edge_count+1
         sndglno((edge_count-1)*sloc+1:edge_count*sloc)= &
           edge_buffer(2:sloc+1,i)
         boundary_ids(edge_count)=edge_buffer(sloc+2,i)
       end if
    end do

    deallocate(edge_buffer)
    deallocate(sndglno)
    deallocate(boundary_ids)

    close(node_unit)

41  continue ! We jump to here if there was no edge file.

    return

42  FLExit("Unable to open "//trim(lfilename)//".node")

43  FLExit("Unable to open "//trim(lfilename)//".ele")


  end function read_triangle_files_to_field_no_faces

  function read_triangle_serial(filename, quad_degree) result (field)

    character(len=*), intent(in) :: filename
    !! The degree of the quadrature.
    integer, intent(in), optional, target :: quad_degree
    !! The degree of the quadrature.

    type(vector_field) :: field
    type(quadrature_type) :: quad
    type(element_type) :: shape

    integer :: dim, loc

    call identify_triangle_file(filename, dim, loc)
    quad=make_quadrature(loc, dim, degree=quad_degree)
    shape=make_element_shape(loc, dim, 1, quad)
    field=read_triangle_files_serial(filename, shape)

    ! deallocate our references of shape and quadrature:
    ! NOTE: we're using the specific deallocate interface here
    !       to make the intel compiler shut up
    call deallocate_element(shape)
    call deallocate(quad)

  end function read_triangle_serial

  function read_triangle_files_serial(filename, shape) result (field)
    !!< Filename is the base name of the triangle file without .node or .ele.

    character(len=*), intent(in) :: filename
    type(element_type), intent(in), target :: shape
    type(vector_field) :: field

    integer :: node_unit, ele_unit
    real, allocatable, dimension(:) :: read_buffer
    integer, allocatable, dimension(:,:) :: edge_buffer
    integer, allocatable, dimension(:) :: sndglno
    integer, allocatable, dimension(:) :: boundary_ids, element_owner

    character(len = parallel_filename_len(filename)) :: lfilename
    integer :: i, j, nodes, dim, xdim, node_attributes, boundaries, &
         ele_attributes, loc, sloc, elements, edges, edge_count
    integer, allocatable, dimension(:):: node_order
    logical :: file_exists
    type(mesh_type) :: mesh

    lfilename = trim(filename)

    node_unit=free_unit()

    ewrite(2, *) "Opening "//trim(lfilename)//".node for reading."
    ! Open node file
    open(unit=node_unit, file=trim(lfilename)//".node", err=42, action="read")

    ! Read node file header.
    read (node_unit, *) nodes, xdim, node_attributes, boundaries

    ele_unit=free_unit()

    ewrite(2, *) "Opening "//trim(lfilename)//".ele for reading."
    ! Open element file
    open(unit=ele_unit, file=trim(lfilename)//".ele", err=43, action="read")

    ! Read element file header.
    read (ele_unit, *) elements, loc, ele_attributes

    assert(loc==shape%ndof)
    allocate(node_order(loc))
    select case(loc)
    case(3)
       node_order = (/1,2,3/)
    case default
       do j = 1, loc
          node_order(j) = j
       end do
    end select

    call allocate(mesh, nodes, elements, shape, name="CoordinateMesh")

    call allocate(field, xdim, mesh, name="Coordinate")

    ! Drop the local reference to mesh - now field owns the only reference.
    call deallocate(mesh)

    allocate(read_buffer(xdim+node_attributes+boundaries+1))

    if(node_attributes==1) then ! this assumes the node attribute are column numbers
       allocate(field%mesh%columns(1:nodes))
    end if

    do i = 1, nodes
       read(node_unit,*) read_buffer
       forall (j=1:xdim)
          field%val(j,i)=read_buffer(j+1)
       end forall
       if (node_attributes==1) then
          field%mesh%columns(i)=floor(read_buffer(xdim+1))
       end if
    end do

    deallocate(read_buffer)
    allocate(read_buffer(loc+ele_attributes+1))

    if(ele_attributes==1) then  ! this assumes that the element attribute is a region id
      call allocate_region_ids(field%mesh, elements)
    end if

    do i = 1, elements
       read(ele_unit,*) read_buffer
       field%mesh%ndglno((i-1)*loc+1:i*loc)=floor(read_buffer(node_order+1))
       if(ele_attributes==1) then
          field%mesh%region_ids(i)=read_buffer(loc+2)
       end if
    end do

    close(node_unit)
    close(ele_unit)

    ! Get the mesh dimension so we know which files to look for
    dim=shape%dim

    ! Open edge file
    select case (dim)
    case(2)
       inquire(file=trim(lfilename)//".edge",exist=file_exists)
       if(file_exists) then
          ewrite(2, *) "Opening "//trim(lfilename)//".edge for reading."
          open(unit=node_unit, file=trim(lfilename)//".edge", err=41, &
               action="read")
       end if
    case(3)
       inquire(file=trim(lfilename)//".face",exist=file_exists)
       if(file_exists) then
          ewrite(2, *) "Opening "//trim(lfilename)//".face for reading."
          open(unit=node_unit, file=trim(lfilename)//".face", err=41, &
               action="read")
       end if
    end select

    if(file_exists) then
       ! Read edge file header.
       read (node_unit, *) edges, boundaries
    else
       edges = 0
       boundaries = 1
    end if

    if(edges==0) then
       file_exists = .false.
       close(node_unit)
    end if

    select case(cell_family(shape))
    case(FAMILY_SIMPLEX)
       if ((loc/=dim+1).and.(boundaries/=0)) then
          ewrite(0,*) "Warning: triangle boundary markers not supported for qua", &
               "dratic space elements."
          if(file_exists) then
             file_exists= .false.
             close(node_unit)
          end if
       end if
       sloc=loc-1
    case default
       FLAbort('Illegal element family')
    end select

    allocate(edge_buffer(sloc+boundaries+1,edges))
    edge_buffer=0
    allocate(sndglno(edges*sloc))
    sndglno=0
    allocate(boundary_ids(1:edges))
    boundary_ids=0
    if (boundaries==2) then
       allocate(element_owner(1:edges))
       element_owner=0
    end if
    edge_count=0

    if (boundaries==0) then
       ewrite(0,*) "Warning: triangle edge file has no boundary markers"
       if(file_exists) then
          file_exists=.false.
          close(node_unit)
       end if
    else
       if(file_exists) then
          read(node_unit, *) edge_buffer
          do i = 1, edges
             if (edge_buffer(sloc+2,i)/=0) then
                ! boundary edge/face
                edge_count=edge_count+1
                sndglno((edge_count-1)*sloc+1:edge_count*sloc)= &
                     edge_buffer(2:sloc+1,i)
                boundary_ids(edge_count)=edge_buffer(sloc+2,i)
                if (boundaries==2) then
                   element_owner(edge_count)=edge_buffer(sloc+3,i)
                end if
             end if
          end do

          file_exists=.false.
          close(node_unit)
       end if
    end if

    if (boundaries<2) then
       call add_faces(field%mesh, &
            sndgln=sndglno(1:edge_count*sloc), &
            boundary_ids=boundary_ids(1:edge_count))
    else
       call add_faces(field%mesh, &
            sndgln=sndglno(1:edge_count*sloc), &
            boundary_ids=boundary_ids(1:edge_count), &
            element_owner=element_owner)
    end if

    deallocate(edge_buffer)
    deallocate(sndglno)
    deallocate(boundary_ids)

41  continue ! We jump to here if there was no edge file.

    return

42  FLExit("Unable to open "//trim(lfilename)//".node")

43  FLExit("Unable to open "//trim(lfilename)//".ele")

  end function read_triangle_files_serial
end module read_triangle
